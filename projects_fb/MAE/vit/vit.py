import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block, _init_vit_weights

from detectron2.modeling import Backbone


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
        self.out_features = out_features
        self.has_cls_embed = has_cls_embed
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        if self.has_cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

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
            )
            self.blocks.append(block)
            name = f"block{i}"

            if name in self.out_features:
                self._out_feature_channels[name] = embed_dim
                self._out_feature_strides[name] = patch_size

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
        outputs = {}
        for i, block in enumerate(self.blocks):
            x = block(x)
            name = f"block{i}"

            if name in self.out_features:
                norm = getattr(self, f"{name}_norm")
                x_out = norm(x)
                if self.has_cls_embed:
                    x_out = x_out[:, 1:]
                outputs[name] = x_out.reshape(bchw[0], bchw[2], bchw[3], -1).permute(0, 3, 1, 2)

        return outputs
