from collections import namedtuple

import torch
from torch import nn

CONV1_OUT_CHANNELS = 64
RES2_OUT_CHANNELS = 256


# ResNet stage specification
StageSpec = namedtuple(
    "StageSpec",
    [
        "index",  # Index of the stage, eg 1, 2, ..,. 5
        "block_count",  # Numer of residual blocks in the stage
        "return_features",  # True => return the last feature map from this stage
    ],
)

# -----------------------------------------------------------------------------
# Standard ResNet models
# -----------------------------------------------------------------------------
# ResNet-50 (including all stages)
ResNet50StagesTo5 = (
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((2, 3, False), (3, 4, False), (4, 6, False), (5, 3, True))
)
# ResNet-50 up to stage 4 (excludes stage 5)
ResNet50StagesTo4 = (
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((2, 3, False), (3, 4, False), (4, 6, True))
)
# ResNet-50-FPN (including all stages)
ResNet50FPNStagesTo5 = (
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((2, 3, True), (3, 4, True), (4, 6, True), (5, 3, True))
)


class FixedBatchNorm2d(nn.Module):
    """Equivalent to AffineChannel in (Caffe2) Detectron."""

    def __init__(self, n):
        super(FixedBatchNorm2d, self).__init__()
        self.register_buffer("scale", torch.zeros(1, n, 1, 1))
        self.register_buffer("bias", torch.zeros(1, n, 1, 1))

    def forward(self, x):
        return x * self.scale + self.bias


# TODO(rbg): add BottleneckWithGroupNorm
class BottleneckWithFixedBatchNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
    ):
        super(BottleneckWithFixedBatchNorm, self).__init__()
        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        self.branch2a = nn.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )
        self.branch2a_bn = FixedBatchNorm2d(bottleneck_channels)
        self.branch2b = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1,
            bias=False,
        )
        self.branch2b_bn = FixedBatchNorm2d(bottleneck_channels)
        self.branch2c = nn.Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )
        self.branch2c_bn = FixedBatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if in_channels != out_channels:
            self.branch1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                groups=num_groups,
            )
            self.branch1_bn = FixedBatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        out = self.branch2a(x)
        out = self.branch2a_bn(out)
        out = self.relu(out)

        out = self.branch2b(out)
        out = self.branch2b_bn(out)
        out = self.relu(out)

        out0 = self.branch2c(out)
        out = self.branch2c_bn(out0)

        if hasattr(self, "branch1"):
            residual = self.branch1(x)
            residual = self.branch1_bn(residual)

        out += residual
        out = self.relu(out)

        return out


# TODO(rbg): Add StemWithGroupNorm
# Why isn't this a module? Because structure is tied to weight serialization
def StemWithFixedBatchNorm(backbone):
    backbone.conv1 = nn.Conv2d(
        3, CONV1_OUT_CHANNELS, kernel_size=7, stride=2, padding=3, bias=False
    )
    backbone.conv1_bn = FixedBatchNorm2d(CONV1_OUT_CHANNELS)
    backbone.relu = nn.ReLU(inplace=True)
    backbone.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


def _make_stage(
    block_module,
    in_channels,
    bottleneck_channels,
    out_channels,
    block_count,
    num_groups,
    stride_in_1x1,
    first_stride,
):
    blocks = []
    stride = first_stride
    for _ in range(block_count):
        blocks.append(
            block_module(
                in_channels,
                bottleneck_channels,
                out_channels,
                num_groups,
                stride_in_1x1,
                stride,
            )
        )
        stride = 1
        in_channels = out_channels
    return nn.Sequential(*blocks)


class ResNetBackbone(nn.Module):
    def __init__(
        self,
        stem_function,
        block_module,
        stages,
        num_groups=1,
        width_per_group=64,
        stride_in_1x1=True,
    ):
        super(ResNetBackbone, self).__init__()

        stem_function(self)

        in_channels = CONV1_OUT_CHANNELS
        stage2_bottleneck_channels = num_groups * width_per_group
        stage2_out_channels = RES2_OUT_CHANNELS
        self.stages = []
        self.return_features = {}
        for stage in stages:
            name = "res" + str(stage.index)
            stage2_relative_factor = 2 ** (stage.index - 2)
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            out_channels = stage2_out_channels * stage2_relative_factor
            module = _make_stage(
                block_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage.block_count,
                num_groups,
                stride_in_1x1,
                first_stride=int(stage.index > 2) + 1,
            )
            in_channels = out_channels
            self.add_module(name, module)
            self.stages.append(name)
            self.return_features[name] = stage.return_features

        # TODO parametrize
        for p in self.conv1.parameters():
            p.requires_grad = False
        for p in self.res2.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outputs = []
        for stage in self.stages:
            x = getattr(self, stage)(x)
            if self.return_features[stage]:
                outputs.append(x)
        return outputs


# TODO: rethink this abstraction
class ResNetHead(nn.Module):
    def __init__(
        self,
        block_module,
        stages,
        num_groups=1,
        width_per_group=64,
        stride_in_1x1=True,
        stride_init=None,
    ):
        in_channels = 1024  # TODO make it generic
        out_channels = 2048  # TODO make it generic
        bottleneck_channels = 512  # TODO make it generic
        super(ResNetHead, self).__init__()

        self.stages = []
        stride = stride_init
        for stage in stages:
            name = "res" + str(stage.index)
            if not stride:
                stride = int(stage.index > 2) + 1
            module = _make_stage(
                block_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage.block_count,
                num_groups,
                stride_in_1x1,
                first_stride=stride,
            )
            stride = None
            self.add_module(name, module)
            self.stages.append(name)

    def forward(self, x):
        for stage in self.stages:
            x = getattr(self, stage)(x)
        return x


# TODO find a better name for this
class ClassifierHead(nn.Module):
    def __init__(self, num_inputs, num_classes):
        super(ClassifierHead, self).__init__()
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.bbox_pred = nn.Linear(num_inputs, num_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.weight, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.weight, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred
