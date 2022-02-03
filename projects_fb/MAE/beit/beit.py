# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
from functools import partial

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
#from timm.models.registry import register_model
from fairscale.nn.checkpoint import checkpoint_wrapper
from detectron2.modeling import Backbone
from scipy import interpolate
import fvcore.nn.weight_init as weight_init

from ..vit.vit import make_window, revert_window
from ..vit.blocks import BasicBlock, BottleneckBlock, SingleBlock, PreBasicBlock, ConvNextBlock
from .attentions import AttentionPartition


logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None, use_cls_token=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
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
                torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_cls_token = use_cls_token

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            if not self.use_cls_token:
                relative_position_bias = relative_position_bias[1:, 1:]
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            if not self.use_cls_token:
                rel_pos_bias = rel_pos_bias[:, 1:, 1:]
            attn = attn + rel_pos_bias
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 rel_window_size=None, attn_head_dim=None, win_size=0, use_cls_token=True,
                 residual_block=None, residual_norm="BN", residual_act="relu", residual_kernel_size=3,
                 residual_num_block=1, residual_drop_path=False,
                 attn="Attention", att_partition_size=49, att_partition_q_shuffle=False,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if attn == "Attention":
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, window_size=rel_window_size, attn_head_dim=attn_head_dim,
                use_cls_token=use_cls_token)
        elif attn == "AttentionPartition":
            assert attn_head_dim is None
            self.attn = AttentionPartition(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, partition_size=att_partition_size, 
                q_shuffle=att_partition_q_shuffle,
            )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if init_values is not None and init_values > 0.:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None
        
        self.win_size = win_size
        self.hw = None

        self.residual_block = residual_block
        if residual_block:
            if residual_block == "basic":
                self.residual = nn.Sequential(
                    *[BasicBlock(
                        in_channels=dim,
                        out_channels=dim,
                        norm=residual_norm,
                        activation=residual_act,
                        kernel_size=residual_kernel_size,
                        drop_path=drop_path if residual_drop_path else 0.0,
                        final_block=(i == residual_num_block - 1),
                    ) for i in range(residual_num_block)]
                )
            elif residual_block == "basic_dw":
                self.residual = nn.Sequential(
                    *[BasicBlock(
                        in_channels=dim,
                        out_channels=dim,
                        norm=residual_norm,
                        activation=residual_act,
                        kernel_size=residual_kernel_size,
                        drop_path=drop_path if residual_drop_path else 0.0,
                        final_block=(i == residual_num_block - 1),
                        num_groups=dim,
                    ) for i in range(residual_num_block)]
                )
            elif residual_block == "bottleneck":
                self.residual = nn.Sequential(
                    *[BottleneckBlock(
                        in_channels=dim,
                        out_channels=dim,
                        bottleneck_channels=dim // 4,
                        norm=residual_norm,
                        activation=residual_act,
                        kernel_size=residual_kernel_size,
                        final_block=(i == residual_num_block - 1),
                    ) for i in range(residual_num_block)]
                )
            elif residual_block == "bottleneck2":
                self.residual = nn.Sequential(
                    *[BottleneckBlock(
                        in_channels=dim,
                        out_channels=dim,
                        bottleneck_channels=dim // 2,
                        norm=residual_norm,
                        activation=residual_act,
                        kernel_size=residual_kernel_size,
                        final_block=(i == residual_num_block - 1),
                    ) for i in range(residual_num_block)]
                )
            elif residual_block == "single":
                self.residual = nn.Sequential(
                    *[SingleBlock(
                        in_channels=dim,
                        out_channels=dim,
                        norm=residual_norm,
                        activation=residual_act,
                        kernel_size=residual_kernel_size,
                        drop_path=drop_path if residual_drop_path else 0.0,
                        final_block=(i == residual_num_block - 1),
                    ) for i in range(residual_num_block)]
                )
            elif residual_block == "single_dw":
                self.residual = nn.Sequential(
                    *[SingleBlock(
                        in_channels=dim,
                        out_channels=dim,
                        norm=residual_norm,
                        activation=residual_act,
                        kernel_size=residual_kernel_size,
                        drop_path=drop_path if residual_drop_path else 0.0,
                        final_block=(i == residual_num_block - 1),
                        num_groups=dim,
                    ) for i in range(residual_num_block)]
                )
            elif residual_block == "pre_basic":
                self.residual = nn.Sequential(
                    *[PreBasicBlock(
                        in_channels=dim,
                        out_channels=dim,
                        norm=residual_norm,
                        activation=residual_act,
                        kernel_size=residual_kernel_size,
                        final_block=(i == residual_num_block - 1),
                    ) for i in range(residual_num_block)]
                )
            elif residual_block == "convnext":
                self.residual = nn.Sequential(
                    *[ConvNextBlock(
                        dim=dim,
                        drop_path=drop_path if residual_drop_path else 0.0,
                        final_block=(i == residual_num_block - 1),
                    ) for i in range(residual_num_block)]
                )
            else:
                raise NotImplementedError

    def forward(self, x, rel_pos_bias=None):
        # if self.gamma_1 is None:
        #     x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
        #     x = x + self.drop_path(self.mlp(self.norm2(x)))
        # else:
        #     x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
        #     x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        ori_x = x
        x = self.norm1(x)
        if self.win_size > 0:
            x, pad_hw = make_window(x, self.hw, self.win_size)
        x = self.attn(x, rel_pos_bias=rel_pos_bias)
        if self.win_size > 0:
            x = revert_window(x, pad_hw, self.hw, self.win_size)
        
        if self.gamma_1 is None:
            x = ori_x + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = ori_x + self.drop_path(self.gamma_1 * x)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))       

        if self.residual_block:
            B, _, C = x.shape
            x = x.reshape(B, self.hw[0], self.hw[1], C).permute(0, 3, 1, 2)

            x = self.residual(x)
            x = x.permute(0, 2, 3, 1).view(B, -1, C)

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        # bchw = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        bchw = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, bchw


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


def resize_res_pos(key, rel_pos_bias, model_rel_pos_bias):
    src_num_pos, num_attn_heads = rel_pos_bias.size()
    dst_num_pos, _ = model_rel_pos_bias.size()
    num_extra_tokens = 3
    src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
    dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
    if src_size != dst_size:
        logger.info("Rel position embed interpolate for %s from %dx%d to %dx%d" % (
            key, src_size, src_size, dst_size, dst_size))
        extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
        rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

        def geometric_progression(a, r, n):
            return a * (1.0 - r ** n) / (1.0 - r)

        left, right = 1.01, 1.5
        while right - left > 1e-6:
            q = (left + right) / 2.0
            gp = geometric_progression(1, q, src_size // 2)
            if gp > dst_size // 2:
                right = q
            else:
                left = q

        # if q > 1.13492:
        #     q = 1.13492

        dis = []
        cur = 1
        for i in range(src_size // 2):
            dis.append(cur)
            cur += q ** (i + 1)

        r_ids = [-_ for _ in reversed(dis)]

        x = r_ids + [0] + dis
        y = r_ids + [0] + dis

        t = dst_size // 2.0
        dx = np.arange(-t, t + 0.1, 1.0)
        dy = np.arange(-t, t + 0.1, 1.0)
        # print("x = {}".format(x))
        # print("dx = {}".format(dx))

        all_rel_pos_bias = []

        for i in range(num_attn_heads):
            z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
            f = interpolate.interp2d(x, y, z, kind='cubic')
            all_rel_pos_bias.append(
                torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

        rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
        rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)

    return rel_pos_bias


class BEiTDet(Backbone):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001, use_cls_token=True,
                 out_block_indexes=[], out_features=[], checkpoint_block_num=0, window_size=0, window_block_indexes=[], use_cls_token_det=True,
                 pretrain_img_size=224, pos_init_checkpoint=None,         
                 residual_block="bottleneck", residual_block_indexes=[], residual_norm="BN", residual_act="relu",
                 residual_kernel_size=3, residual_num_block=1, residual_drop_path=False,
                 att_partition_block_indexes=[], att_partition_size=49, att_partition_q_shuffle=False,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self._out_features = out_features
        if len(out_block_indexes) == 0:
            assert len(out_features) != 0
            # remove 'block' prefix
            out_block_indexes = [int(name[5:]) for name in out_features]

        self._out_block_indexes = out_block_indexes

        self.use_cls_token = use_cls_token
        self.use_cls_token_det = use_cls_token_det
        self.window_block_indexes = window_block_indexes

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        # num_patches = self.patch_embed.num_patches

        if use_cls_token_det:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            # assert use_cls_token_det  # we don't have this version yet
            # assert len(window_block_indexes) == 0 # not implement yet
            if len(window_block_indexes) > 0:
                self.rel_pos_bias = RelativePositionBias(window_size=(window_size, window_size), num_heads=num_heads)
                if len(window_block_indexes) < depth:
                    self.rel_pos_bias_global = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
                else:
                    self.rel_pos_bias_global = None
            else:
                self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
                self.rel_pos_bias_global = None
        else:
            self.rel_pos_bias = None
            self.rel_pos_bias_global = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList()
        self._out_feature_channels = {}
        self._out_feature_strides = {}
        for i in range(depth):
            block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, rel_window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                win_size=window_size if i in window_block_indexes else 0, use_cls_token=use_cls_token_det,
                residual_block=None if i not in residual_block_indexes else residual_block,
                residual_norm=residual_norm,
                residual_act=residual_act,
                residual_kernel_size=residual_kernel_size,
                residual_num_block=residual_num_block,
                residual_drop_path=residual_drop_path,
                attn="AttentionPartition" if i in att_partition_block_indexes else "Attention",
                att_partition_size=att_partition_size,
                att_partition_q_shuffle=att_partition_q_shuffle,
            )
            if i + 1 <= checkpoint_block_num:
                block = checkpoint_wrapper(block)

            self.blocks.append(block)
            # name = f"block{i}"
            for block_idx, name in zip(self._out_block_indexes, self._out_features):
                if block_idx == i:
                    self._out_feature_channels[name] = embed_dim
                    self._out_feature_strides[name] = patch_size

        # self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        # self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        # trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if pos_init_checkpoint is not None:
            self._load_pos_weights(pos_init_checkpoint)

        # self.head.weight.data.mul_(init_scale)
        # self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if hasattr(m, "final_linear") and m.final_linear:
                print("final_linear")
                nn.init.constant_(m.weight, 0)
            else:
                trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 0.0 if hasattr(m, "final_norm") and m.final_norm else 1.0)
            if hasattr(m, "final_norm") and m.final_norm:
                print("final_norm")
        elif isinstance(m, nn.Conv2d):
            # residual blocks
            if hasattr(m, "final_conv") and m.final_conv:
                print("final_conv")
                nn.init.constant_(m.weight, 0.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            else:
                weight_init.c2_msra_fill(m)
        
        if hasattr(m, "final_norm") or hasattr(m, "final_linear") or hasattr(m, "final_conv"):
            nn.init.constant_(m.weight, 0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            print("final: zero_init")
    
    def _load_pos_weights(self, pos_init_checkpoint):
        checkpoint = torch.load(pos_init_checkpoint, map_location=torch.device("cpu"))
        assert "model" in checkpoint
        state_dict = checkpoint["model"]

        # abs pos embed
        if "pos_embed" in state_dict and hasattr(self, "pos_embed"):
            pos_embed_checkpoint = state_dict['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = self.patch_embed.num_patches
            num_extra_tokens = self.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # # height (== width) for the new position embedding
            # new_size = int(num_patches ** 0.5)
            new_size = self.patch_embed.patch_shape
            # class_token and dist_token are kept unchanged
            if orig_size != new_size[0] or orig_size != new_size[1]:
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size[0], new_size[1]), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                pos_embed_checkpoint = torch.cat((extra_tokens, pos_tokens), dim=1)

                logger.info(f"Resize pos embed from checkpoint from {orig_size}x{orig_size} to {new_size}")
            
            self.pos_embed.data[...] = pos_embed_checkpoint.clone()
            logger.info("Init pos embed from checkpoint")
                
        # rel pos bias
        for key, param in self.named_parameters():
            if "relative_position_bias_table" in key:
                ckpt_key = key.replace("_global", "") if "rel_pos_bias_global" in key else key
                if ckpt_key in state_dict:
                    rel_pos_bias = state_dict[ckpt_key]
                    new_rel_pos_bias = resize_res_pos(key, rel_pos_bias, param).clone()
                    param.data[...] = new_rel_pos_bias
                    logger.info(f"Init {key} from {ckpt_key} in checkpoint")

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _get_pos_embed(self, pos_embed, bchw):
        h, w = bchw[-2], bchw[-1]
        if self.use_cls_token:
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
        if self.use_cls_token_det:
            pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)

        return pos_embed

    def forward(self, x):
        x, bchw = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self._get_pos_embed(self.pos_embed, bchw)
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if len(self.window_block_indexes) > 0:
            rel_pos_bias_global = self.rel_pos_bias_global() if self.rel_pos_bias_global is not None else None

        outputs = {}
        for i, blk in enumerate(self.blocks):
            blk.hw = (bchw[2], bchw[3])
            bias = rel_pos_bias
            if len(self.window_block_indexes) > 0 and i not in self.window_block_indexes:
                bias = rel_pos_bias_global
            x = blk(x, rel_pos_bias=bias)
            # name = f"block{i}"

            if i in self._out_block_indexes:
                x_out = x
                if self.cls_token is not None:
                    x_out = x_out[:, 1:]
                x_out = x_out.reshape(bchw[0], bchw[2], bchw[3], -1).permute(0, 3, 1, 2)
                for block_idx, name in zip(self._out_block_indexes, self._out_features):
                    if block_idx == i:
                        outputs[name] = x_out

        return outputs


# @register_model
# def beit_base_patch16_224(pretrained=False, **kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     return model


# @register_model
# def beit_base_patch16_384(pretrained=False, **kwargs):
#     model = VisionTransformer(
#         img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     return model


# @register_model
# def beit_large_patch16_224(pretrained=False, **kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     return model


# @register_model
# def beit_large_patch16_384(pretrained=False, **kwargs):
#     model = VisionTransformer(
#         img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     return model


# @register_model
# def beit_large_patch16_512(pretrained=False, **kwargs):
#     model = VisionTransformer(
#         img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     return model