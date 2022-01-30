import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, Mlp, DropPath
from timm.models.vision_transformer import _init_vit_weights
from fairscale.nn.checkpoint import checkpoint_wrapper

from detectron2.modeling import Backbone

from .blocks import LayerNorm


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        bchw = x.shape
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, bchw


def make_window(x, hw, win_size):
    # print(x.shape)
    B, _, C = x.shape
    H, W = hw
    x = x.view(B, H, W, C)

    pad_h = (win_size - H % win_size) % win_size
    pad_w = (win_size - W % win_size) % win_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    Hp, Wp = H + pad_h, W + pad_w

    # B, H, W, C -> B * nWin, win_size, win_size, C --> B * nWin, win_size * win_size, C
    # print(x.shape, hw, Hp, Wp)
    x = x.view(B, Hp // win_size, win_size, Wp // win_size, win_size, C)
    # print(x.shape)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size * win_size, C)
    
    return x, (Hp, Wp)


def revert_window(x, pad_hw, hw, win_size):
    Hp, Wp = pad_hw
    H, W = hw
    B = x.shape[0] // (Hp * Wp // win_size // win_size)
   
    # B * nWin, win_size, win_size, C -> B, H, W, C 
    x = x.view(B, Hp // win_size, Wp // win_size, win_size, win_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    
    x = x.view(B, H * W, -1)

    return x


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_cls_token=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.use_cls_token = use_cls_token

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if rel_pos_bias is not None:
            if not self.use_cls_token:
                rel_pos_bias = rel_pos_bias[:, 1:, 1:]
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, win_size=0, use_cls_token=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, use_cls_token=use_cls_token
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.win_size = win_size
        self.hw = None

    def forward(self, x, rel_pos_bias=None):
        ori_x = x
        x = self.norm1(x)
        if self.win_size > 0:
            x, pad_hw = make_window(x, self.hw, self.win_size)
        x = self.attn(x, rel_pos_bias=rel_pos_bias)
        if self.win_size > 0:
            x = revert_window(x, pad_hw, self.hw, self.win_size)

        x = ori_x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
            
        return x


class VisionTransformerDet(Backbone):
    """Vision Transformer Backbone for detection
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        out_features=None,
        has_cls_embed=False,
        checkpoint_block_num=0,
        window_size=14,
        window_block_indexes=[],
        use_shared_rel_pos_bias=False,
        pretrain_img_size=224,
        out_norm=True,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        # self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self._out_features = out_features
        self.has_cls_embed = has_cls_embed
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.window_block_indexes = window_block_indexes
        self.out_norm = out_norm

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        # num_patches = self.patch_embed.num_patches

        if self.has_cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            # assert use_cls_token_det  # we don't have this version yet
            # assert len(window_block_indexes) == 0 # not implement yet
            if len(window_block_indexes) > 0:
                self.rel_pos_bias = RelativePositionBias(window_size=(window_size, window_size), num_heads=num_heads)
                if len(window_block_indexes) < depth:
                    self.rel_pos_bias_global = RelativePositionBias(window_size=self.patch_embed.grid_size, num_heads=num_heads)
                else:
                    self.rel_pos_bias_global = None
            else:
                self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.grid_size, num_heads=num_heads)
                self.rel_pos_bias_global = None
        else:
            self.rel_pos_bias = None
            self.rel_pos_bias_global = None

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList()
        self._out_feature_channels = {}
        self._out_feature_strides = {}
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                win_size=window_size if i in window_block_indexes else 0,
                use_cls_token=has_cls_embed,
            )
            if i + 1 <= checkpoint_block_num:
                block = checkpoint_wrapper(block)

            self.blocks.append(block)
            name = f"block{i}"

            if name in self._out_features:
                self._out_feature_channels[name] = embed_dim
                self._out_feature_strides[name] = patch_size

                if out_norm:
                    layer = norm_layer(embed_dim)
                    self.add_module(f"{name}_norm", layer)

        # self.norm = norm_layer(embed_dim)

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        # trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_vit_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def _get_pos_embed(self, pos_embed, bchw):
        h, w = bchw[-2], bchw[-1]
        cls_pos_embed = pos_embed[:, 0:1, :]
        pos_embed = pos_embed[:, 1:]
        xy_num = pos_embed.shape[1]
        # print(xy_num, h, w, h * w)
        if xy_num != h * w:

            size = int(math.sqrt(xy_num))
            assert size * size == xy_num
            new_pos_embed = F.interpolate(
                pos_embed.reshape(1, size, size, -1).permute(0, 3, 1, 2),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )

            pos_embed = new_pos_embed.reshape(1, -1, h * w).permute(0, 2, 1)
            if self.has_cls_embed:
                pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)

        return pos_embed

    def forward(self, x):
        x, bchw = self.patch_embed(x)
        if self.has_cls_embed:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self._get_pos_embed(self.pos_embed, bchw))

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if len(self.window_block_indexes) > 0:
            rel_pos_bias_global = self.rel_pos_bias_global() if self.rel_pos_bias_global is not None else None

        outputs = {}
        for i, block in enumerate(self.blocks):
            block.hw = (bchw[2], bchw[3])
            bias = rel_pos_bias
            if len(self.window_block_indexes) > 0 and i not in self.window_block_indexes:
                bias = rel_pos_bias_global
            x = block(x, rel_pos_bias=bias)
            name = f"block{i}"

            if name in self._out_features:
                if self.out_norm:
                    norm = getattr(self, f"{name}_norm")
                    x_out = norm(x)
                else:
                    x_out = x
                if self.has_cls_embed:
                    x_out = x_out[:, 1:]
                outputs[name] = x_out.reshape(bchw[0], bchw[2], bchw[3], -1).permute(0, 3, 1, 2)

        return outputs


class ViTUp(Backbone):
    def __init__(self, net, in_features, scale_factors):
        super(ViTUp, self).__init__()
        assert isinstance(net, Backbone)
        assert len(in_features) == len(scale_factors)

        self.scale_factors = scale_factors

        input_shapes = net.output_shape()
        self.net = net
        self._out_features = in_features
        self._out_feature_channels = {f: input_shapes[f].channels for f in in_features}
        self._out_feature_strides = {
            f: int(input_shapes[f].stride / scale) for f, scale in zip(in_features, scale_factors)
        }
        print(self._out_feature_channels, self._out_feature_strides)

    def forward(self, x):
        features = self.net(x)

        outputs = {}
        for f, scale in zip(self._out_features, self.scale_factors):
            outputs[f] = F.interpolate(features[f], scale_factor=scale, mode="bilinear")

        return outputs


class ViTUp1(Backbone):
    def __init__(self, net, in_features, scale_factors, embed_dim, mode=1, resize_ratio=1.0):
        super(ViTUp1, self).__init__()
        assert isinstance(net, Backbone)
        assert len(in_features) == len(scale_factors)

        self.scale_factors = scale_factors
        self.resize_ratio = resize_ratio

        input_shapes = net.output_shape()
        self.net = net
        self._out_features = in_features
        self._out_feature_channels = {f: input_shapes[f].channels for f in in_features}
        self._out_feature_strides = {
            f: int(input_shapes[f].stride / resize_ratio / scale) for f, scale in zip(in_features, scale_factors)
        }
        print("resize", self._out_feature_channels, self._out_feature_strides, input_shapes)
        for i, scale in enumerate(scale_factors):
            if scale == 4.0:
                if mode == 1:
                    layer = nn.Sequential(
                        nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                        nn.SyncBatchNorm(embed_dim),
                        nn.GELU(),
                        nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                    )
                elif mode == 2:
                    layer = nn.Sequential(
                        nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                        nn.ReLU(),
                        nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                    ) 
                elif mode == 3:
                    layer = nn.Sequential(
                        nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                        nn.GELU(),
                        nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                    ) 
                elif mode == 4:
                    layer = nn.Sequential(
                        nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                        nn.SyncBatchNorm(embed_dim),
                        nn.ReLU(),
                        nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                    )
                elif mode == 5:
                    layer = nn.Sequential(
                        nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                        nn.GroupNorm(32, embed_dim),
                        nn.GELU(),
                        nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                    )
                elif mode == 6:
                    layer = nn.Sequential(
                        nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                        LayerNorm(embed_dim, data_format="channels_first"),
                        nn.GELU(),
                        nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                    )
                elif mode == 7:
                    layer = nn.Sequential(
                        nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                        LayerNorm(embed_dim, data_format="channels_first"),
                        nn.GELU(),
                        nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                        LayerNorm(embed_dim, data_format="channels_first"),
                        nn.GELU(),
                    )
                else:
                    raise NotImplementedError
            elif scale == 2.0:
                if mode != 7:
                    layer = nn.Sequential(
                        nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                    )
                else:
                    layer = nn.Sequential(
                        nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                        LayerNorm(embed_dim, data_format="channels_first"),
                        nn.GELU(),
                    )
                   
            elif scale == 1.0:
                layer = nn.Identity()
            elif scale == 0.5:
                layer = nn.MaxPool2d(kernel_size=2, stride=2)
            
            self.add_module(f"stage_{i}", layer)

    def forward(self, x):
        features = self.net(x)

        outputs = {}
        for i in range(len(self.scale_factors)):
            f = features[self._out_features[i]]
            if self.resize_ratio != 1.0:
                f = F.interpolate(f, scale_factor=self.resize_ratio, mode="bicubic")
            layer = getattr(self, f"stage_{i}")
            outputs[self._out_features[i]] = layer(f)

        return outputs