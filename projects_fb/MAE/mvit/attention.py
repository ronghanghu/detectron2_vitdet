#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

# from slowfast.models.common import DropPath, Mlp

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop_rate=0.0,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        if self.drop_rate > 0.0:
            self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        x = self.fc2(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        return x


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def attention_pool(tensor, pool, thw_shape, has_cls_embed=True, norm=None):
    if pool is None:
        return tensor, thw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    T, H, W = thw_shape
    tensor = (
        tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
    )

    tensor = pool(tensor)

    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    if norm is not None:
        tensor = norm(tensor)
    # Assert tensor_dim in [3, 4]
    if tensor_dim == 4:
        pass
    else:  #  tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, thw_shape


def get_rel_pos(rel_pos, d):
    if isinstance(d, int):
        ori_d = rel_pos.shape[0]
        if ori_d == d:
            return rel_pos
        else:
            # Interpolate rel pos.
            new_pos_embed = F.interpolate(
                rel_pos.reshape(1, ori_d, -1).permute(0, 2, 1),
                size=d,
                mode="linear",
            )

            return new_pos_embed.reshape(-1, d).permute(1, 0)


def cal_rel_pos_spatial(attn, q, has_cls_embed, q_shape, k_shape, rel_pos_h, rel_pos_w):
    sp_idx = 1 if has_cls_embed else 0
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape
    dh = int(2 * max(q_h, k_h) - 1)
    dw = int(2 * max(q_w, k_w) - 1)
    # Intepolate rel pos if needed.
    rel_pos_h = get_rel_pos(rel_pos_h, dh)
    rel_pos_w = get_rel_pos(rel_pos_w, dw)

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = torch.arange(q_h)[:, None] * q_h_ratio- torch.arange(k_h)[None, :] * k_h_ratio
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio
    dist_w += (k_w - 1) * k_w_ratio

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_t, q_h, q_w, dim)
    # [B, H, q_t, q_h, q_w, dim] -> [q_h, B, H, q_t, q_w, dim] -> [q_h, B*H*q_t*q_w, dim]
    r_q_h = r_q.permute(3, 0, 1, 2, 4, 5).reshape(q_h, B * n_head * q_t * q_w, dim)
    # [B, H, q_t, q_h, q_w, dim] -> [q_w, B, H, q_t, q_h, dim] -> [q_w, B*H*q_t*q_h, dim]
    r_q_w = r_q.permute(4, 0, 1, 2, 3, 5).reshape(q_w, B * n_head * q_t * q_h, dim)

    # [q_h, B*H*q_t*q_w, dim] * [q_h, dim, k_h] = [q_h, B*H*q_t*q_w, k_h] -> [B*H*q_t*q_w, q_h, k_h]
    rel_h = torch.matmul(r_q_h, Rh.permute(0, 2, 1)).transpose(0, 1)
    # [q_w, B*H*q_t*q_h, dim] * [q_w, dim, k_w] = [q_w, B*H*q_t*q_h, k_w] -> [B*H*q_t*q_h, q_w, k_w]
    rel_w = torch.matmul(r_q_w, Rw.permute(0, 2, 1)).transpose(0, 1)

    # [B*H*q_t*q_w, q_h, k_h] -> [B, H, q_t, qh, qw, k_h]
    rel_h = rel_h.view(B, n_head, q_t, q_w, q_h, k_h).permute(0, 1, 2, 4, 3, 5)
    # [B*H*q_t*q_h, q_w, k_w] -> [B, H, q_t, qh, qw, k_w]
    rel_w = rel_w.view(B, n_head, q_t, q_h, q_w, k_w)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_t, q_h, q_w, k_t, k_h, k_w)
        + rel_h[:, :, :, :, :, None, :, None]
        + rel_w[:, :, :, :, :, None, None, :]
    ).view(B, -1, q_t * q_h * q_w, k_t * k_h * k_w)

    return attn


def cal_rel_pos_temporal(attn, q, has_cls_embed, q_shape, k_shape, rel_pos_t):
    sp_idx = 1 if has_cls_embed else 0
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape
    dt = int(2 * max(q_t, k_t) - 1)
    # Intepolate rel pos if needed.
    rel_pos_t = get_rel_pos(rel_pos_t, dt)

    # Scale up rel pos if shapes for q and k are different.
    q_t_ratio = max(k_t / q_t, 1.0)
    k_t_ratio = max(q_t / k_t, 1.0)
    dist_t = torch.arange(q_t)[:, None] * q_t_ratio- torch.arange(k_t)[None, :] * k_t_ratio
    dist_t += (k_t - 1) * k_t_ratio
    Rt = rel_pos_t[dist_t.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_t, q_h, q_w, dim)
    # [B, H, q_t, q_h, q_w, dim] -> [q_t, B, H, q_h, q_w, dim] -> [q_t, B*H*q_h*q_w, dim]
    r_q = r_q.permute(2, 0, 1, 3, 4, 5).reshape(q_t, B * n_head * q_h * q_w, dim)

    # [q_t, B*H*q_h*q_w, dim] * [q_t, dim, k_t] = [q_t, B*H*q_h*q_w, k_t] -> [B*H*q_h*q_w, q_t, k_t]
    rel = torch.matmul(r_q, Rt.transpose(1, 2)).transpose(0, 1)
    # [B*H*q_h*q_w, q_t, k_t] -> [B, H, q_t, q_h, q_w, k_t]
    rel = rel.view(B, n_head, q_h, q_w, q_t, k_t).permute(0, 1, 4, 2, 3, 5)

    #attn[:, :, 1:, 1:] += attn_t
    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_t, q_h, q_w, k_t, k_h, k_w)
        + rel[:, :, :, :, :, :, None, None]
    ).view(B, -1, q_t * q_h * q_w, k_t * k_h * k_w)

    return attn


def window_partition(x, window_size):
    """
    Args:
        x: (B, T, H, W, C)
        window_size : (w_T, w_H, w_W)
    Returns:
        windows: (num_windows*B, w_T, w_H, w_W, C)
    """
    B, T, H, W, C = x.shape
    wT, wH, wW = window_size
    x = x.view(B, T // wT, wT, H // wH, wH, W // wW, wW, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, wT, wH, wW, C)
    return windows


def window_reverse(windows, ori_size):
    """
    Args:
        windows: (num_windows*B, w_T, w_H, w_W, C)
        ori_size: (T, H, W)
    Returns:
        x: (B, T, H, W, C)
    """
    wT, wH, wW = windows.shape[1:4]
    T, H, W = ori_size #int(ori_size[0]), int(ori_size[1]), int(ori_size[2])
    B = windows.shape[0] // (T * H * W // (wT * wH * wW))
    x = windows.view(B, T // wT, H // wH, W // wW, wT, wH, wW, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, T, H, W, -1)
    return x


def make_window(x, thw, win_size):
    #print(x.shape)
    B, N, _, C = x.shape
    #C = x.shape[-1]
    T, H, W = thw
    x_wins = x.contiguous().view(-1, T, H, W, C)
    pads = [(w - x % w) % w for x, w in zip(thw, win_size)]

    if sum(pads) > 0:
        x_wins = F.pad(x_wins, (0, 0, 0, pads[2], 0, pads[1], 0, pads[0]))

    # -> B * N * nW, w1, w2, w3, C
    x_wins = window_partition(x_wins, win_size)
    x_wins = x_wins.view(B, N, -1, win_size[0] * win_size[1] * win_size[2], C)
    # --> B * nw, N, _, C
    x_wins = x_wins.transpose(1, 2).reshape(-1, N, win_size[0] * win_size[1] * win_size[2], C)

    return x_wins, pads


def revert_window(x, thw, pads, win_size):
    _, _, C = x.shape
    #l_pads, r_pads = pads
    x = x.view(-1, win_size[0], win_size[1], win_size[2], x.shape[-1])
    # x = x.view(-1, self.attn.thw_out[0], self.attn.thw_out[1], self.attn.thw_out[2], x_block.shape[-1])
    # ratios = [att_out / att_in for att_in, att_out in zip(self.attn.thw, self.attn.thw_out)]
    # self.thw_out = [math.ceil(d * r) for d, r in zip(self.thw, ratios)]
    # pad_out = [math.ceil(d * r) for d, r in zip(pad_thw, ratios)]
    # shift_out = [math.ceil(s * r) for s, r in zip(self.shift_size, ratios)]
    pad_out = [k + p for k, p in zip(thw, pads)]
    # print("revert", x.shape, pad_out)
    x = window_reverse(x, pad_out)

    x = x[:, :thw[0], :thw[1], :thw[2], :].contiguous()

    x = x.view(-1, thw[0] * thw[1] * thw[2], C)

    return x


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim,
        input_size,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        # Options include `conv`, `avg`, and `max`.
        mode="conv",
        # If True, perform pool before projection.
        pool_first=False,
        rel_pos_spatial=False,
        rel_pos_temporal=False,
        win_size=False,
    ):
        super().__init__()
        self.pool_first = pool_first
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        self.win_size = win_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.has_cls_embed = has_cls_embed
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        assert not (self.win_size  and self.has_cls_embed)

        if pool_first:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.k = nn.Linear(dim, dim, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if numpy.prod(kernel_q) == 1 and numpy.prod(stride_q) == 1:
            kernel_q = ()
        if numpy.prod(kernel_kv) == 1 and numpy.prod(stride_kv) == 1:
            kernel_kv = ()

        if mode == "avg":
            self.pool_q = (
                nn.AvgPool3d(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            self.pool_k = (
                nn.AvgPool3d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                nn.AvgPool3d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
        elif mode == "max":
            self.pool_q = (
                nn.MaxPool3d(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            self.pool_k = (
                nn.MaxPool3d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                nn.MaxPool3d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
        elif mode == "conv":
            self.pool_q = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_q) > 0
                else None
            )
            self.norm_q = norm_layer(head_dim) if len(kernel_q) > 0 else None
            self.pool_k = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_k = norm_layer(head_dim) if len(kernel_kv) > 0 else None
            self.pool_v = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_v = norm_layer(head_dim) if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

        self.rel_pos_spatial = rel_pos_spatial
        self.rel_pos_temporal = rel_pos_temporal
        if self.rel_pos_spatial:
            assert input_size[1] == input_size[2]
            q_size = input_size[1] // stride_q[1] if len(stride_q) > 0 else input_size[1]
            kv_size = input_size[1] // stride_kv[1] if len(stride_kv) > 0 else input_size[1]
            rel_sp_dim = 2 * max(q_size, kv_size) - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            trunc_normal_(self.rel_pos_h, std=0.02)
            trunc_normal_(self.rel_pos_w, std=0.02)

        if self.rel_pos_temporal:
            self.rel_pos_t = nn.Parameter(torch.zeros(2 * input_size[0] - 1, dim // num_heads))
            trunc_normal_(self.rel_pos_t, std=0.02)

        if self.win_size:
            self.stride_q = stride_q if len(stride_q) > 0 else [1, 1, 1]
            self.stride_kv = stride_kv if len(stride_kv) > 0 else [1, 1, 1]
            self.q_win_size = [w // s for w, s in zip(self.win_size, self.stride_q)]
            self.kv_win_size = [w // s for w, s in zip(self.win_size, self.stride_kv)]
            print(self.stride_q, self.stride_kv, self.win_size, self.q_win_size, self.kv_win_size)

    def forward(self, x, thw_shape):
        # print(x.shape)
        B, N, C = x.shape
        if self.pool_first:
            x = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q = k = v = x
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

        q, q_shape = attention_pool(
            q,
            self.pool_q,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
        )
        k, k_shape = attention_pool(
            k,
            self.pool_k,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )
        v, v_shape = attention_pool(
            v,
            self.pool_v,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )

        if self.pool_first:
            if self.has_cls_embed:
                q_N = numpy.prod(q_shape) + 1
                k_N = numpy.prod(k_shape) + 1
                v_N = numpy.prod(v_shape) + 1
            else:
                q_N = numpy.prod(q_shape)
                k_N = numpy.prod(k_shape)
                v_N = numpy.prod(v_shape)

            # print(q.shape, B, q_N, C, q_shape)
            q = q.permute(0, 2, 1, 3).reshape(B, q_N, C)
            q = self.q(q).reshape(B, q_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            v = v.permute(0, 2, 1, 3).reshape(B, v_N, C)
            v = self.v(v).reshape(B, v_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            k = k.permute(0, 2, 1, 3).reshape(B, k_N, C)
            k = self.k(k).reshape(B, k_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.win_size:
            q, q_pads = make_window(q, q_shape, self.q_win_size)
            k, k_pads = make_window(k, k_shape, self.kv_win_size)
            v, v_pads = make_window(v, v_shape, self.kv_win_size)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.rel_pos_spatial:
            attn = cal_rel_pos_spatial(
                attn, q, self.has_cls_embed, q_shape, k_shape, self.rel_pos_h, self.rel_pos_w
            )

        if self.rel_pos_temporal:
            attn = cal_rel_pos_temporal(
                attn, q, self.has_cls_embed, q_shape, k_shape, self.rel_pos_t,
            )

        attn = attn.softmax(dim=-1)

        N = q.shape[2]
        x = (attn @ v).transpose(1, 2).reshape(-1, N, C)

        if self.win_size:
            x = revert_window(x, q_shape, q_pads, self.q_win_size)

        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)
        return x, q_shape


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        input_size,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        up_rate=None,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        mode="conv",
        has_cls_embed=True,
        pool_first=False,
        rel_pos_spatial=False,
        rel_pos_temporal=False,
        win_size=False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.win_size = win_size
        self.norm1 = norm_layer(dim)
        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        self.attn = MultiScaleAttention(
            dim,
            num_heads=num_heads,
            input_size=input_size,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=nn.LayerNorm,
            has_cls_embed=has_cls_embed,
            mode=mode,
            pool_first=pool_first,
            rel_pos_spatial=rel_pos_spatial,
            rel_pos_temporal=rel_pos_temporal,
            win_size=win_size,
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        # TODO: check the use case for up_rate, and merge the following lines
        if up_rate is not None and up_rate > 1:
            mlp_dim_out = dim * up_rate
        else:
            mlp_dim_out = dim_out
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
            act_layer=act_layer,
            drop_rate=drop_rate,
        )
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        self.pool_skip = (
            nn.MaxPool3d(
                kernel_skip, stride_skip, padding_skip, ceil_mode=False
            )
            if len(kernel_skip) > 0
            else None
        )

    def forward(self, x, thw_shape):
        x_block, thw_shape_new = self.attn(self.norm1(x), thw_shape)
        x_res, _ = attention_pool(
            x, self.pool_skip, thw_shape, has_cls_embed=self.has_cls_embed
        )
        x = x_res + self.drop_path(x_block)
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        if self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)
        return x, thw_shape_new
    
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        
        return self
