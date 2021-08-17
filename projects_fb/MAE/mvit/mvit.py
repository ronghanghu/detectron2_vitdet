# Copyright (c) Facebook, Inc. and its affiliates.

import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from .attention import MultiScaleBlock

from detectron2.layers import Conv2d, FrozenBatchNorm2d, get_norm
from detectron2.modeling import BACKBONE_REGISTRY, ResNet, ResNetBlockBase, Backbone
from detectron2.modeling.backbone.resnet import BasicStem, BottleneckBlock, DeformBottleneckBlock
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from detectron2.layers import ShapeSpec

#try:
from fairscale.nn.checkpoint import checkpoint_wrapper
# except ImportError:
#     checkpoint_wrapper = None


class PatchEmbed(nn.Module):
    """
    PatchEmbed.
    """

    def __init__(
        self,
        dim_in=3,
        dim_out=768,
        kernel=(1, 16, 16),
        stride=(1, 4, 4),
        padding=(1, 7, 7),
        conv_2d=False,
    ):
        super().__init__()
        if conv_2d:
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d
        self.proj = conv(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.proj(x)
        # B C (T) H W -> B (T)HW C
        return x.flatten(2).transpose(1, 2), x.shape
    
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        
        return self


def round_width(width, multiplier, min_width=1, divisor=1, verbose=False):
    if not multiplier:
        return width
    width *= multiplier
    min_width = min_width or divisor
    # if verbose:
    #     logger.info(f"min width {min_width}")
    #     logger.info(f"width {width} divisor {divisor}")
    #     logger.info(f"other {int(width + divisor / 2) // divisor * divisor}")

    width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)


class MViT(Backbone):
    """
    Multiscale Vision Transformers
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, cfg, in_chans):
        super().__init__()
        # Get parameters.
       # assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg
        pool_first = cfg.MVIT.POOL_FIRST
        # Prepare input.
        spatial_size = cfg.MVIT.IMAGE_SIZE
        temporal_size = 1

        use_2d_patch = cfg.MVIT.PATCH_2D
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        # Prepare output.
        embed_dim = cfg.MVIT.EMBED_DIM
        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS
        self.drop_rate = cfg.MVIT.DROPOUT_RATE
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        # Params for positional embedding
        self.use_abs_pos = cfg.MVIT.USE_ABS_POS
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        self.rel_pos_spatial = cfg.MVIT.REL_POS_SPATIAL
        self.rel_pos_temporal = cfg.MVIT.REL_POS_TEMPORAL
        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.out_features = cfg.MVIT.OUT_FEATURES

        self.patch_embed = PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=use_2d_patch,
        )
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        num_patches = math.prod(self.patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.use_abs_pos:
            if self.sep_pos_embed:
                self.pos_embed_spatial = nn.Parameter(
                    torch.zeros(
                        1, self.patch_dims[1] * self.patch_dims[2], embed_dim
                    )
                )
                self.pos_embed_temporal = nn.Parameter(
                    torch.zeros(1, self.patch_dims[0], embed_dim)
                )
                if self.cls_embed_on:
                    self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
            else:
                self.pos_embed = nn.Parameter(
                    torch.zeros(1, pos_embed_dim, embed_dim)
                )

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

        pool_q = [[] for i in range(cfg.MVIT.DEPTH)]
        pool_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        win_sizes = [False] * cfg.MVIT.DEPTH

        for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
            stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][
                1:
            ]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s for s in cfg.MVIT.POOL_Q_STRIDE[i][1:]
                ]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE is not None:
            _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
            cfg.MVIT.POOL_KV_STRIDE = []
            for i in range(cfg.MVIT.DEPTH):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
            stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[
                i
            ][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_kv[
                    cfg.MVIT.POOL_KV_STRIDE[i][0]
                ] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in cfg.MVIT.POOL_KV_STRIDE[i][1:]
                ]

        for k in cfg.MVIT.WIN_SIZE_STRIDE:
            win_sizes[k[0]] = [w // s for w, s in zip(cfg.MVIT.WIN_SIZE, k[1:])]
        # print(cfg.MVIT.WIN_SIZE_STRIDE)
        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None

        input_size = self.patch_dims
        self.blocks = nn.ModuleList()
        self._out_feature_strides = {}
        self._out_feature_channels = {}
        self._stages = {}
        stage = 1
        stride = cfg.MVIT.PATCH_STRIDE[-1]
        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            embed_dim = round_width(embed_dim, dim_mul[i], divisor=num_heads)
            dim_out = round_width(
                embed_dim,
                dim_mul[i + 1],
                divisor=round_width(num_heads, head_mul[i + 1]),
            )
            # print(win_sizes[i])
            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                input_size=input_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=mode,
                has_cls_embed=False,  # detection don't need cls embed
                pool_first=pool_first,
                rel_pos_spatial=self.rel_pos_spatial,
                rel_pos_temporal=self.rel_pos_temporal,
                win_size=win_sizes[i],
            )
            if cfg.MVIT.ACT_CHECKPOINT and checkpoint_wrapper is not None:
                print("checkpointing")
                block = checkpoint_wrapper(block)

            self.blocks.append(block)

            if len(stride_q[i]) > 0:
                input_size = [size // stride for size, stride in zip(input_size, stride_q[i])]
                stride *= stride_q[i][-1]
                # strides = [s * q_s for s, q_s in zip(strides, pool_q[i])]

            if i == depth - 1 or dim_mul[i + 1] == 2:
                stage += 1
                name = f'scale{stage}'
                self._stages[i] = stage
                # print(i, stage, name, self.out_features)
                if name in self.out_features:
                    print("output_features", name, i, dim_out, stride)
                    self._out_feature_channels[name] = dim_out
                    self._out_feature_strides[name] = stride

                    layer = norm_layer(dim_out)
                    self.add_module(f"{name}_norm", layer)
            
            name = f"block{i}"
            if name in self.out_features:
                print("output_features", name, i, dim_out, stride)
                self._out_feature_channels[name] = dim_out
                self._out_feature_strides[name] = stride

                layer = norm_layer(dim_out)
                self.add_module(f"{name}_norm", layer)

        embed_dim = dim_out
        # self.norm = norm_layer(embed_dim)

        # self.head = head_helper.TransformerBasicHead(
        #     embed_dim,
        #     num_classes,
        #     dropout_rate=cfg.MODEL.DROPOUT_RATE,
        #     act_func=cfg.MODEL.HEAD_ACT,
        # )
        if self.use_abs_pos:
            if self.sep_pos_embed:
                trunc_normal_(self.pos_embed_spatial, std=0.02)
                trunc_normal_(self.pos_embed_temporal, std=0.02)
                if self.cls_embed_on:
                    trunc_normal_(self.pos_embed_class, std=0.02)
            else:
                trunc_normal_(self.pos_embed, std=0.02)
        # if self.cls_embed_on:
        #     trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        self.freeze(cfg.MVIT.FREEZE_AT, cfg.MVIT.FREEZE_POS)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     names = []
    #     if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
    #         if self.use_abs_pos:
    #             if self.sep_pos_embed:
    #                 names.extend(
    #                     ["pos_embed_spatial", "pos_embed_temporal", "pos_embed_class"]
    #                 )
    #             else:
    #                 names.append(["pos_embed"])
    #         if self.rel_pos_spatial:
    #             names.extend(["rel_pos_h", "rel_pos_w"])
    #         if self.rel_pos_temporal:
    #             names.extend(["rel_pos_t"])
    #         if self.cls_embed_on:
    #             names.append("cls_token")

    #     return names

    def _get_pos_embed(self, pos_embed, bchw):
        # print(bchw)
        h, w = bchw[-2], bchw[-1]
        if self.cls_embed_on:
            cls_pos_embed = pos_embed[:, 0:1, :]
            pos_embed = pos_embed[:, 1:]
        xy_num = pos_embed.shape[1]
        # print(xy_num, h, w, h * w)
        if xy_num != h * w:
            size = int(math.sqrt(xy_num))
            assert size * size == xy_num
            # cls_pos_embed = pos_embed[:, 0:1, :]
            new_pos_embed = F.interpolate(
                pos_embed[:, :, :].reshape(1, size, size, -1).permute(0, 3, 1, 2),
                size=(h, w),
                mode="bicubic",
            )

            pos_embed = new_pos_embed.reshape(1, -1, h * w).permute(0, 2, 1)

        if self.cls_embed_on:
            pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)

        # print(self.cls_embed_on, pos_embed.shape)

        return pos_embed

    def forward(self, x):
        # x = x[0]
        x, bchw = self.patch_embed(x)
        T, H, W = 1, bchw[-2], bchw[-1]

        # T = self.cfg.DATA.NUM_FRAMES // self.patch_stride[0]
        # H = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[1]
        # W = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[2]
        B, N, C = x.shape

        # detection don't need cls token
        # if self.cls_embed_on:
        #     cls_tokens = self.cls_token.expand(
        #         B, -1, -1
        #     )  # stole cls_tokens impl from Phil Wang, thanks
        #     x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            # if self.sep_pos_embed:
            #     pos_embed = self.pos_embed_spatial.repeat(
            #         1, self.patch_dims[0], 1
            #     ) + torch.repeat_interleave(
            #         self.pos_embed_temporal,
            #         self.patch_dims[1] * self.patch_dims[2],
            #         dim=1,
            #     )
            #     if self.cls_embed_on:
            #         pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
            #     x = x + pos_embed
            # else:
            pos_embed = self._get_pos_embed(
                self.pos_embed,
                bchw,
            )
            # print(x.shape, pos_embed.shape, self.cls_embed_on)
            x = x + (pos_embed[:, 1:] if self.cls_embed_on else pos_embed)

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        # x = self.norm(x)
        # if self.cls_embed_on:
        #     x = x[:, 0]
        # else:
        #     x = x.mean(1)

        # x = self.head(x)
        outputs = {}
        thw = [T, H, W]
        for i, blk in enumerate(self.blocks):
            x, thw = blk(x, thw)

            if i in self._stages:
                name = f"scale{self._stages[i]}"
                if name in self.out_features:
                    norm = getattr(self, f"{name}_norm")
                    x_out = norm(x)
                    #if self.cls_embed_on:
                    #    x_out = x_out[:, 1:]
                    #print(x_out.shape, B, self.thw)
                    outputs[name] = x_out.reshape(B, thw[1], thw[2], -1).permute(0, 3, 1, 2)
            
            name = f"block{i}"
            if name in self.out_features:
                norm = getattr(self, f"{name}_norm")
                x_out = norm(x)
                outputs[name] = x_out.reshape(B, thw[1], thw[2], -1).permute(0, 3, 1, 2)

        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self.out_features
        }
    
    def freeze(self, freeze_at=0, freeze_pos=False):
        if freeze_pos and self.use_abs_pos:
            print("freeze pos")
            if self.sep_pos_embed:
                self.pos_embed_spatial.require_grad = False
                self.pos_embed_temporal.require_grad = False
                if self.cls_embed_on:
                    self.pos_embed_class.require_grad = False
            else:
                self.pos_embed.require_grad = False

        if freeze_at >= 1:
            print("freeze patch_embed")
            self.patch_embed.freeze()
            if self.norm_stem:
                for p in self.norm_stem.parameters():
                    p.require_grad = False
        
        for idx, block in enumerate(self.blocks, start=2):
            if freeze_at >= idx:
                print(f"freeze block {idx - 2}")
                block.freeze()
            else:
                break
        
        return self


# @BACKBONE_REGISTRY.register()
# def build_mvit_backbone(cfg, input_shape):
#     return MViT(cfg, in_chans=input_shape.channels)


# @BACKBONE_REGISTRY.register()
# def build_mvit_fpn_backbone(cfg, input_shape):
#     """
#     Args:
#         cfg: a detectron2 CfgNode

#     Returns:
#         backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
#     """
#     bottom_up = build_mvit_backbone(cfg, input_shape)
#     in_features = cfg.MODEL.FPN.IN_FEATURES
#     out_channels = cfg.MODEL.FPN.OUT_CHANNELS
#     backbone = FPN(
#         bottom_up=bottom_up,
#         in_features=in_features,
#         out_channels=out_channels,
#         norm=cfg.MODEL.FPN.NORM,
#         top_block=LastLevelMaxPool(),
#         fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
#     )
#     return backbone


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
