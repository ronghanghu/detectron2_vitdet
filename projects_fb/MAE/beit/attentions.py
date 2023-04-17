import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import get_norm

# modified from https://github.com/endernewton/mae/blob/km/video_exp/util/video_vit.py


def partition_attention(q, k, v, heads, scale, q_shuffle=False, partition_size=49):
    """
    q: [N, T, S, D] or [N, T*S, D]
    k, v: [N, T, R, D] or [N, T*R, D]
    TODO: speed to be optimized
    """
    q_shape = q.shape
    N, D = q.shape[0], q.shape[-1]
    q = q.view([N, -1, D])
    k = k.view([N, -1, D])
    v = v.view([N, -1, D])

    L = q.shape[1]
    # shuffle k, v
    with torch.no_grad():
        noise = torch.rand(N, L, device=q.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=-1)  # ascend: small is keep, large is remove
    if q_shuffle:
        ids_rev_shuffle = torch.argsort(ids_shuffle, dim=-1)
        ids_rev_shuffle = ids_rev_shuffle.unsqueeze(-1).repeat(1, 1, D)

    ids_shuffle = ids_shuffle.unsqueeze(-1).repeat(1, 1, D)
    if q_shuffle:
        q = torch.gather(q, dim=1, index=ids_shuffle)
    k = torch.gather(k, dim=1, index=ids_shuffle)
    v = torch.gather(v, dim=1, index=ids_shuffle)

    assert L % partition_size == 0

    q = k.view([-1, partition_size, heads, D // heads])
    k = k.view([-1, partition_size, heads, D // heads])
    v = v.view([-1, partition_size, heads, D // heads])

    # in this einsum:
    # h: heads
    # q: L of q
    # k: L of k
    attn = torch.einsum('nqhd,nkhd->nqkh', q, k)  #
    attn *= scale
    attn = attn.softmax(dim=-2)  # along the axis of L of k

    # in this einsum:
    # h: heads
    # q: L of q
    # k: L of k (same as L of v)
    x = torch.einsum('nqkh,nkhd->nqhd', attn, v)
    if q_shuffle:
        # reverse q shuffle
        x = x.reshape([N, -1, D])
        x = torch.gather(x, dim=1, index=ids_rev_shuffle)

    x = x.reshape(q_shape)  # => [N, T, S, D] or [N, T*S, D]
    # x = x.flatten(-2)  # => [N, T, S, D]
    return x


class AttentionPartition(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., q_shuffle=False, partition_size=49,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        assert attn_drop == 0.  # do not use
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.requires_t_shape = False  # requires temporal shape

        self.partition_size = partition_size
        self.q_shuffle = q_shuffle

    def forward(self, x, rel_pos_bias=None):
        assert rel_pos_bias is None
        N, L, D = x.shape

        qkv = self.qkv(x)  # [N, T, S, 3 * D]
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.split(D, dim=-1)

        x = partition_attention(q, k, v, heads=self.num_heads, scale=self.scale, partition_size=self.partition_size, q_shuffle=self.q_shuffle)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def spatiotemporal_attention(q, k, v, heads, scale):
    """
    q: [N, T, S, D] or [N, T*S, D]
    k, v: [N, T, R, D] or [N, T*R, D]
    TODO: speed to be optimized
    """
    q_shape = q.shape
    N, D = q.shape[0], q.shape[-1]
    q = q.view([N, -1, heads, D // heads])
    k = k.view([N, -1, heads, D // heads])
    v = v.view([N, -1, heads, D // heads])

    # in this einsum:
    # h: heads
    # q: L of q
    # k: L of k
    attn = torch.einsum('nqhd,nkhd->nqkh', q, k)  # [N, L, L, heads]
    attn *= scale
    attn = attn.softmax(dim=-2)  # along the axis of L of k

    # in this einsum:
    # h: heads
    # q: L of q
    # k: L of k (same as L of v)
    x = torch.einsum('nqkh,nkhd->nqhd', attn, v)
    x = x.reshape(q_shape)  # => [N, T, S, D] or [N, T*S, D]
    # x = x.flatten(-2)  # => [N, T, S, D]
    return x


# this is a wrapper
class AttentionSubsample(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        assert attn_drop==0.  # do not use
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.requires_t_shape = True  # requires temporal shape

   # @staticmethod
    def subsample_func(self, y):
        raise NotImplementedError

    def forward(self, x, rel_pos_bias=None):
        assert rel_pos_bias is None
        N, S, D = x.shape  # S: H*W

        qkv = self.qkv(x)  # [N, T, S, 3 * D]
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)

        q, k, v = qkv.split(D, dim=-1)

        k = self.subsample_func(k)
        v = self.subsample_func(v)

        x = spatiotemporal_attention(q, k, v, heads=self.num_heads, scale=self.scale)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionSubsampleMaxpool(AttentionSubsample):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_kernel=2):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)

        self.pool_kernel = pool_kernel
    # max pool subsample
    # @staticmethod
    def subsample_func(self, y):
        N, S, D = y.shape  # S: H*W
        # spatial subsampling by max pooling
        # note: pool3d works for [N, C, T, H, W] order, here we have the [N, T, H, W, C] tensor.
        # we adpat the kernel accordingly, as we only pool H and W
        kernel_size = [self.pool_kernel, self.pool_kernel, 1]
        y = torch.nn.functional.max_pool3d(
                y.reshape([N, 1, int(S**.5), -1, D]),
                kernel_size=kernel_size, stride=kernel_size
            ).view([N, -1, D])
        return y


def attention_pool(tensor, pool, norm=None, use_cls_token=False):
    if use_cls_token:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    H = int(L**.5)
    assert H * H == L
    tensor = tensor.reshape(B * N, H, H, C).permute(0, 3, 1, 2).contiguous()
    tensor = pool(tensor)

    tensor = tensor.reshape(B, N, C, -1).transpose(2, 3)

    if use_cls_token:
        tensor = torch.cat((cls_tok, tensor), dim=2)

    if norm is not None:
        tensor = norm(tensor)

    return tensor


class ResidualMaxConv(nn.Module):
    def __init__(self, dim, pool_kernel, pool_stride, groups=1):
        super().__init__()

        self.max_pool = nn.MaxPool2d(pool_kernel, pool_stride, pool_kernel // 2, ceil_mode=False)
        self.conv_pool = nn.Conv2d(
            dim, dim, pool_kernel, stride=pool_stride,
            padding=pool_kernel // 2, groups=groups,
        )
        self.conv_pool_norm = get_norm("LN", dim)

        self.conv_pool_norm.final_norm = True

    def forward(self, x):
        # print(x.shape, self.max_pool(x).shape, self.conv_pool(x).shape)
        return self.max_pool(x) + self.conv_pool_norm(self.conv_pool(x))


class AttentionPool(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None, use_cls_token=True,
            pool_mode="max", pool_kernel=3, pool_stride=2):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

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

        if pool_mode == "max":
            self.pool_k = nn.MaxPool2d(pool_kernel, pool_stride, pool_kernel // 2, ceil_mode=False)
            self.pool_v = nn.MaxPool2d(pool_kernel, pool_stride, pool_kernel // 2, ceil_mode=False)
        elif pool_mode == "conv" or pool_mode == "convnorm":
            self.pool_k = nn.Conv2d(
                head_dim, head_dim, pool_kernel, stride=pool_stride,
                padding=pool_kernel // 2, groups=head_dim,
            )
            self.pool_v = nn.Conv2d(
                head_dim, head_dim, pool_kernel, stride=pool_stride,
                padding=pool_kernel // 2, groups=head_dim,
            )
            if pool_mode == "convnorm":
                self.pool_norm_k = nn.LayerNorm(head_dim, eps=1e-6)
                self.pool_norm_v = nn.LayerNorm(head_dim, eps=1e-6)
        elif pool_mode == "convnorm_full":
            self.pool_k = nn.Conv2d(
                head_dim, head_dim, pool_kernel, stride=pool_stride,
                padding=pool_kernel // 2,
            )
            self.pool_v = nn.Conv2d(
                head_dim, head_dim, pool_kernel, stride=pool_stride,
                padding=pool_kernel // 2,
            )
            self.pool_norm_k = nn.LayerNorm(head_dim, eps=1e-6)
            self.pool_norm_v = nn.LayerNorm(head_dim, eps=1e-6)
        elif pool_mode == "max_convnorm":
            self.pool_k = ResidualMaxConv(head_dim, pool_kernel, pool_stride, groups=head_dim)
            self.pool_v = ResidualMaxConv(head_dim, pool_kernel, pool_stride, groups=head_dim)
        elif pool_mode == "max_convnorm_full":
            self.pool_k = ResidualMaxConv(head_dim, pool_kernel, pool_stride, groups=1)
            self.pool_v = ResidualMaxConv(head_dim, pool_kernel, pool_stride, groups=1)
        else:
            raise NotImplementedError

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        k = attention_pool(k, self.pool_k, getattr(self, "pool_norm_k", None), self.use_cls_token)
        v = attention_pool(v, self.pool_v, getattr(self, "pool_norm_v", None), self.use_cls_token)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        assert self.relative_position_bias_table is None and rel_pos_bias is None

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
