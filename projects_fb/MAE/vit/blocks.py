# Copyright (c) Facebook, Inc. and its affiliates.
import fvcore.nn.weight_init as weight_init
from torch import nn
import torch
import torch.nn.functional as F

from detectron2.layers import (
    CNNBlockBase,
    get_norm,
)
from detectron2.layers import get_norm as get_norm_default
from timm.models.layers import DropPath


def get_norm(norm, out_channels):
    if norm == "LN":
        return LayerNorm(out_channels, data_format="channels_first")
    else:
        return get_norm_default(norm, out_channels)


class SingleBlock(CNNBlockBase):
    """
    The basic residual block for ResNet-18 and ResNet-34 defined in :paper:`ResNet`,
    with two 3x3 conv layers and a projection shortcut if needed.
    """

    def __init__(self, in_channels, out_channels, *, stride=1, kernel_size=3, norm="BN", activation="relu", final_block=True, drop_path=0.0, num_groups=1):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first conv.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, stride)

        if activation == "relu":
            act_layer = nn.ReLU
        elif activation == "gelu":
            act_layer = nn.GELU
        else:
            raise NotImplementedError

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
            self.shortcut_norm = get_norm(norm, out_channels)
        else:
            self.shortcut = None

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=False,
        )
        self.conv1_norm = get_norm(norm, out_channels)
        self.act1 = act_layer() if not final_block else nn.Identity()

        for layer in [self.conv1, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        if self.conv1_norm:
            self.conv1_norm.final_norm = final_block
        else:
            self.conv1.final_conv = final_block

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_norm(out) if self.conv1_norm else out

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out = shortcut + self.drop_path(out)
        out = self.act1(out)
        return out


class BasicBlock(CNNBlockBase):
    """
    The basic residual block for ResNet-18 and ResNet-34 defined in :paper:`ResNet`,
    with two 3x3 conv layers and a projection shortcut if needed.
    """

    def __init__(self, in_channels, out_channels, *, stride=1, kernel_size=3, norm="BN", activation="relu", final_block=True, drop_path=0.0, num_groups=1):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first conv.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, stride)

        if activation == "relu":
            act_layer = nn.ReLU
        elif activation == "gelu":
            act_layer = nn.GELU
        else:
            raise NotImplementedError

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
            self.shortcut_norm = get_norm(norm, out_channels)
        else:
            self.shortcut = None

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=False,
        )
        self.conv1_norm = get_norm(norm, out_channels)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=False,
        )
        self.conv2_norm = get_norm(norm, out_channels)
        self.act2 = act_layer() if not final_block else nn.Identity()

        for layer in [self.conv1, self.conv2, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        if self.conv2_norm:
            self.conv2_norm.final_norm = final_block
        else:
            self.conv2.final_conv = final_block

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_norm(out) if self.conv1_norm else out
        out = self.act1(out)

        out = self.conv2(out)
        out = self.conv2_norm(out) if self.conv2_norm else out

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out = shortcut + self.drop_path(out)
        out = self.act2(out)
        return out





class PreBasicBlock(CNNBlockBase):
    """
    The basic residual block for ResNet-18 and ResNet-34 defined in :paper:`ResNet`,
    with two 3x3 conv layers and a projection shortcut if needed.
    """

    def __init__(self, in_channels, out_channels, *, stride=1, kernel_size=3, norm="BN", activation="relu", final_block=True):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first conv.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, stride)

        if activation == "relu":
            act_layer = nn.ReLU
        elif activation == "gelu":
            act_layer = nn.GELU
        else:
            raise NotImplementedError

        assert in_channels == out_channels

        self.conv1_norm = get_norm(norm, out_channels)
        self.act1 = act_layer()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )

        self.conv2_norm = get_norm(norm, out_channels)
        self.act2 = act_layer()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )

        for layer in [self.conv1, self.conv2]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        self.conv2.final_conv = final_block

    def forward(self, x):
        out = self.conv1_norm(x) if self.conv1_norm else x
        out = self.act1(out)
        out = self.conv1(out)

        out = self.conv2_norm(out) if self.conv2_norm else out
        out = self.act2(out)
        out = self.conv2(out)

        out += x
        return out


class BottleneckBlock(CNNBlockBase):
    """
    The standard bottleneck residual block used by ResNet-50, 101 and 152
    defined in :paper:`ResNet`.  It contains 3 conv layers with kernels
    1x1, 3x3, 1x1, and a projection shortcut if needed.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        kernel_size=3,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
        activation="relu",
        final_block=True,
    ):
        """
        Args:
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            num_groups (int): number of groups for the 3x3 conv layer.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            stride_in_1x1 (bool): when stride>1, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
            dilation (int): the dilation rate of the 3x3 conv layer.
        """
        super().__init__(in_channels, out_channels, stride)

        if activation == "relu":
            act_layer = nn.ReLU
        elif activation == "gelu":
            act_layer = nn.GELU
        else:
            raise NotImplementedError

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
            self.shortcut_norm = get_norm(norm, out_channels)
        else:
            self.shortcut = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = nn.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )
        self.conv1_norm = get_norm(norm, bottleneck_channels)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=kernel_size,
            stride=stride_3x3,
            padding=kernel_size // 2 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
        )
        self.conv2_norm = get_norm(norm, bottleneck_channels)
        self.act2 = act_layer()

        self.conv3 = nn.Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.conv3_norm = get_norm(norm, out_channels)
        self.act3 = act_layer() if not final_block else nn.Identity()

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        if self.conv3_norm:
            self.conv3_norm.final_norm = final_block
        else:
            self.conv3_norm.final_conv = final_block

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_norm(out) if self.conv1_norm else out
        out = self.act1(out)

        out = self.conv2(out)
        out = self.conv2_norm(out) if self.conv2_norm else out
        out = self.act2(out)

        out = self.conv3(out)
        out = self.conv3_norm(out) if self.conv3_norm else out

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = self.act3(out)
        return out


class ConvNextBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1.0, final_block=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value >= 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.pwconv2.final_linear = final_block

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm1(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Conv31Block(nn.Module):
    r"""
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0., final_block=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim) # depthwise conv
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.act = nn.GELU()
        self.pwconv1 = nn.Linear(dim, dim) # pointwise/1x1 convs, implemented with linear layers
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.pwconv1.final_linear = final_block

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm1(x)
        x = self.act(x)
        x = self.pwconv1(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Conv31Group32Block(nn.Module):
    r"""
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0., final_block=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=32) # depthwise conv
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.act = nn.GELU()
        self.pwconv1 = nn.Linear(dim, dim) # pointwise/1x1 convs, implemented with linear layers
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.pwconv1.final_linear = final_block

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm1(x)
        x = self.act(x)
        x = self.pwconv1(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MaxPoolBlock(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.pool = self.max_pool = nn.MaxPool2d(kernel_size, 1, padding)

    def forward(self, x):
        return self.pool(x)

        return x


class AvgPoolBlock(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.pool = nn.AvgPool2d(kernel_size, 1, padding)

    def forward(self, x):
        return self.pool(x)

        return x
