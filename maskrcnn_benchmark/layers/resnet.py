from abc import ABCMeta, abstractmethod

import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm2d

from . import Conv2d, FrozenBatchNorm2d


__all__ = ["ResNetBlockBase", "BottleneckBlock", "BasicStem", "ResNet", "make_stage"]


def _get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable):
    Returns:
        nn.Module: the normalization layer
    """
    if isinstance(norm, str):
        norm = {"BN": BatchNorm2d, "FrozenBN": FrozenBatchNorm2d}[norm]
    return norm(out_channels)


class ResNetBlockBase(nn.Module, metaclass=ABCMeta):
    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        return self

    @abstractmethod
    def forward(self):
        pass


class BottleneckBlock(ResNetBlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False
    ):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN"}).
            stride_in_1x1 (bool): when stride==2, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _get_norm(norm, out_channels),
            )
        else:
            self.downsample = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels, bottleneck_channels, kernel_size=1, stride=stride_1x1, bias=False
        )
        # TODO maybe change name if using GN
        self.bn1 = _get_norm(norm, bottleneck_channels)

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1,
            bias=False,
            groups=num_groups,
        )
        self.bn2 = _get_norm(norm, bottleneck_channels)

        self.conv3 = Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = _get_norm(norm, out_channels)

        # TODO initialization

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out += residual
        out = F.relu_(out)
        return out


def make_stage(block_class, num_blocks, first_stride, **kwargs):
    """
    Create a resnet stage by creating many blocks.
    Args:
        block_class (class): a subclass of ResNetBlockBase
        num_blocks (int):
        first_stride (int): the stride of the first block. The other blocks will have stride=1.
            A `stride` argument will be passed to the block constructor.
        kwargs: other arguments passed to the block constructor.

    Returns:
        list[nn.Module]: a list of block module.
    """
    blocks = []
    for i in range(num_blocks):
        blocks.append(block_class(stride=first_stride if i == 0 else 1, **kwargs))
        kwargs["in_channels"] = kwargs["out_channels"]
    return blocks


class BasicStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, norm="BN"):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN"}).
        """
        super().__init__()
        self.conv1 = Conv2d(
            in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = _get_norm(norm, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


class ResNet(nn.Module):
    def __init__(self, stem, stages, num_classes=None, return_features=None):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[ResNetBlock]]): several (typically 4) stages,
                each contains multiple :class:`ResNetBlockBase`.
            num_classes (None or int): if None, will not perform classification.
            return_features (iterable[str]): name of the layers whose outputs should be returned in forward.
                Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
                Note that the outputs will always be returned in the same order as they are in the model,
                regardless of their order in `return_features`.
        """
        super(ResNet, self).__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = 4  # we assume stem has stride 4
        self._feature_strides = {"stem": current_stride}

        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            for block in blocks:
                assert isinstance(block, ResNetBlockBase), block
                curr_channels = block.out_channels
            stage = nn.Sequential(*blocks)
            name = "res" + str(i + 2)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)
            nn.init.normal_(self.linear.weight, stddev=0.01)
            name = "linear"

        if return_features is None:
            return_features = [name]
        self.return_features = set(return_features)
        assert len(self.return_features)
        children = [x[0] for x in self.named_children()]
        for return_feature in self.return_features:
            assert return_feature in children, "Available children: {}".format(", ".join(children))

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        if "stem" in self.return_features:
            outputs.append(x)
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self.return_features:
                outputs.append(x)
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = self.linear(x)
            if "linear" in self.return_features:
                outputs.append(x)
        return outputs

    @property
    def feature_strides(self):
        """
        Returns a dict containing strides of each featuremap.
        The key is the name of the featuremap.
        Could be one of "stem", "res2", ..., "res5".
        """
        return self._feature_strides
