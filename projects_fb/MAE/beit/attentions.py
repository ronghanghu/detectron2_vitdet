import torch
import torch.nn as nn
import torch.nn.functional as F

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

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)

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

        # qkv = self.qkv(x)  # [N, T, S, 3 * D]

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)

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

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)


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

        # qkv = self.qkv(x)  # [N, T, S, 3 * D]

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
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


# class AttentionSubsampleConv(AttentionSubsample):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_kernel=2):
#         super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)

#         stride = pool_kernel
#         kernel = pool_kernel + 1
#         pad = kernel // 2
#         self.pool = nn.Conv2d(
#             dim, dim, kernel_size=(kernel, kernel), stride=(stride, stride),
#             padding=(pad, pad), groups=dim
#         )

#     # max pool subsample
#     # @staticmethod
#     def subsample_func(self, y):
#         N, S, D = y.shape  # S: H*W
#         # spatial subsampling by max pooling
#         # note: pool3d works for [N, C, T, H, W] order, here we have the [N, T, H, W, C] tensor.
#         # we adpat the kernel accordingly, as we only pool H and W
#         # kernel_size = [self.pool_kernel, self.pool_kernel]
#         # y = torch.nn.functional.max_pool2d(
#         #         y.reshape([N, int(S**.5), -1, D]),
#         #         kernel_size=kernel_size, stride=kernel_size
#         #     ).view([N, -1, D])

#         return y