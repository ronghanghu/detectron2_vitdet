# -*- coding: utf-8 -*-

from detectron2.layers import ShapeSpec
from detectron2.modeling import FPN, ResNet, RetinaNet
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone.fpn import LastLevelP6P7
from detectron2.modeling.backbone.resnet import BasicStem, BottleneckBlock
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.meta_arch.retinanet import RetinaNetHead

from newconfig import LazyCall as L

model = L(RetinaNet)(
    backbone=L(FPN)(
        bottom_up=L(ResNet)(
            stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
            # can create some specializations such as "ResNet50" to avoid writing this
            stages=[
                L(ResNet.make_stage)(
                    block_class=BottleneckBlock,
                    num_blocks=n,
                    stride_per_block=[s] + [1] * (n - 1),
                    in_channels=i,
                    bottleneck_channels=o // 4,
                    out_channels=o,
                    stride_in_1x1=True,
                    norm="FrozenBN",
                )
                for (n, s, i, o) in zip(
                    [3, 4, 6, 3], [1, 2, 2, 2], [64, 256, 512, 1024], [256, 512, 1024, 2048]
                )
            ],
            out_features=["res3", "res4", "res5"],
        ),
        in_features=["res3", "res4", "res5"],
        out_channels=256,
        top_block=L(LastLevelP6P7)(in_channels=2048, out_channels="${..out_channels}"),
    ),
    head=L(RetinaNetHead)(
        input_shape=[ShapeSpec(channels=256)],
        num_classes="${..num_classes}",
        conv_dims=[256, 256, 256, 256],
        prior_prob=0.01,
        num_anchors=9,
    ),
    anchor_generator=L(DefaultAnchorGenerator)(
        sizes=[[x, x * 2 ** (1.0 / 3), x * 2 ** (2.0 / 3)] for x in [32, 64, 128, 256, 512]],
        aspect_ratios=[0.5, 1.0, 2.0],
        strides=[8, 16, 32, 64, 128],
        offset=0.0,
    ),
    box2box_transform=L(Box2BoxTransform)(weights=[1.0, 1.0, 1.0, 1.0]),
    anchor_matcher=L(Matcher)(
        thresholds=[0.4, 0.5], labels=[0, -1, 1], allow_low_quality_matches=True
    ),
    num_classes=80,
    head_in_features=["p3", "p4", "p5", "p6", "p7"],
    focal_loss_alpha=0.25,
    focal_loss_gamma=2.0,
    pixel_mean=[103.530, 116.280, 123.675],
    pixel_std=[1.0, 1.0, 1.0],
    input_format="BGR",
)
