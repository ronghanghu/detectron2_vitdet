from detectron2.layers import ShapeSpec
from detectron2.modeling import FPN, GeneralizedRCNN, ResNet, StandardROIHeads
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.backbone.resnet import BasicStem, BottleneckBlock
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.rpn import RPN, StandardRPNHead
from detectron2.modeling.roi_heads import FastRCNNOutputLayers, MaskRCNNConvUpsampleHead
from detectron2.modeling.roi_heads.box_head import FastRCNNConvFCHead

from newconfig import Lazy as D

model = D(
    GeneralizedRCNN,
    backbone=D(
        FPN,
        bottom_up=D(
            ResNet,
            stem=D(BasicStem, in_channels=3, out_channels=64),
            # can create some specializations such as "ResNet50" to avoid writing this
            stages=[
                D(
                    ResNet.make_stage,
                    block_class=BottleneckBlock,
                    num_blocks=n,
                    stride_per_block=[s] + [1] * (n - 1),
                    in_channels=i,
                    bottleneck_channels=o // 4,
                    out_channels=o,
                    stride_in_1x1=True,
                )
                for (n, s, i, o) in zip(
                    [3, 4, 6, 3], [1, 2, 2, 2], [64, 256, 512, 1024], [256, 512, 1024, 2048]
                )
            ],
            out_features=["res2", "res3", "res4", "res5"],
            # freeze_at=2,
        ),
        in_features=["res2", "res3", "res4", "res5"],
        out_channels=256,
        top_block=D(LastLevelMaxPool),
    ),
    proposal_generator=D(
        RPN,
        in_features=["p2", "p3", "p4", "p5", "p6"],
        head=D(StandardRPNHead, in_channels=256, num_anchors=3),
        anchor_generator=D(
            DefaultAnchorGenerator,
            sizes=[[32], [64], [128], [256], [512]],
            aspect_ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            offset=0.0,
        ),
        anchor_matcher=D(
            Matcher, thresholds=[0.3, 0.7], labels=[0, -1, 1], allow_low_quality_matches=True
        ),
        box2box_transform=D(Box2BoxTransform, weights=[1.0, 1.0, 1.0, 1.0]),
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_topk=(2000, 1000),
        post_nms_topk=(1000, 1000),
        nms_thresh=0.7,
    ),
    roi_heads=D(
        StandardROIHeads,
        num_classes=80,
        batch_size_per_image=512,
        positive_fraction=0.25,
        proposal_matcher=D(
            Matcher, thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
        ),
        box_in_features=["p2", "p3", "p4", "p5"],
        box_pooler=D(
            ROIPooler,
            output_size=7,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        box_head=D(
            FastRCNNConvFCHead,
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[],
            fc_dims=[1024, 1024],
        ),
        box_predictor=D(
            FastRCNNOutputLayers,
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.05,
            box2box_transform=D(Box2BoxTransform, weights=(10, 10, 5, 5)),
            # NOTE: interpolation supported by OmegaConf
            num_classes="${..num_classes}",
        ),
        mask_in_features=["p2", "p3", "p4", "p5"],
        mask_pooler=D(
            ROIPooler,
            output_size=14,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        mask_head=D(
            MaskRCNNConvUpsampleHead,
            input_shape=ShapeSpec(channels=256, width=14, height=14),
            num_classes="${..num_classes}",
            conv_dims=[256, 256, 256, 256, 256],
        ),
    ),
    pixel_mean=[103.530, 116.280, 123.675],
    pixel_std=[1.0, 1.0, 1.0],
    input_format="BGR",
)
