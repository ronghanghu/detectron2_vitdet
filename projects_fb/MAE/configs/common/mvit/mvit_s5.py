from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator import RPN, StandardRPNHead
from detectron2.modeling.roi_heads import (
    StandardROIHeads,
    FastRCNNOutputLayers,
    MaskRCNNConvUpsampleHead,
    FastRCNNConvFCHead,
)

from .mvit_config import config as cfg
from ....mvit.mvit import MViT


cfg.MVIT.PATCH_2D = True
cfg.MVIT.MODE = "conv"
cfg.MVIT.CLS_EMBED_ON = False
cfg.MVIT.PATCH_KERNEL = [7, 7]
cfg.MVIT.PATCH_STRIDE = [4, 4]
cfg.MVIT.PATCH_PADDING = [3, 3]
cfg.MVIT.DROPPATH_RATE = 0.2
cfg.MVIT.DEPTH = 16
cfg.MVIT.DIM_MUL = [[1, 2.0], [3, 2.0], [14, 2.0]]
cfg.MVIT.HEAD_MUL = [[1, 2.0], [3, 2.0], [14, 2.0]]
cfg.MVIT.POOL_KVQ_KERNEL = [1, 3, 3]
cfg.MVIT.POOL_KV_STRIDE = [
    [0, 1, 4, 4],
    [1, 1, 2, 2],
    [2, 1, 2, 2],
    [3, 1, 1, 1],
    [4, 1, 1, 1],
    [5, 1, 1, 1],
    [6, 1, 1, 1],
    [7, 1, 1, 1],
    [8, 1, 1, 1],
    [9, 1, 1, 1],
    [10, 1, 1, 1],
    [11, 1, 1, 1],
    [12, 1, 1, 1],
    [13, 1, 1, 1],
    [14, 1, 1, 1],
    [15, 1, 1, 1],
]
cfg.MVIT.POOL_Q_STRIDE = [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]]
cfg.MVIT.OUT_FEATURES = ["scale5"]


model = L(GeneralizedRCNN)(
    backbone=L(MViT)(cfg=cfg, in_chans=3),
    proposal_generator=L(RPN)(
        in_features=["scale5"],
        head=L(StandardRPNHead)(in_channels=768, num_anchors=15),
        anchor_generator=L(DefaultAnchorGenerator)(
            sizes=[[32, 64, 128, 256, 512]],
            aspect_ratios=[0.5, 1.0, 2.0],
            strides=[16],
            offset=0.0,
        ),
        anchor_matcher=L(Matcher)(
            thresholds=[0.3, 0.7], labels=[0, -1, 1], allow_low_quality_matches=True
        ),
        box2box_transform=L(Box2BoxTransform)(weights=[1.0, 1.0, 1.0, 1.0]),
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_topk=(12000, 6000),
        post_nms_topk=(2000, 1000),
        nms_thresh=0.7,
    ),
    roi_heads=L(StandardROIHeads)(
        num_classes=80,
        batch_size_per_image=512,
        positive_fraction=0.25,
        proposal_matcher=L(Matcher)(
            thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
        ),
        box_in_features="${..proposal_generator.in_features}",
        box_pooler=L(ROIPooler)(
            output_size=7,
            scales=(1.0 / 16,),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        box_head=L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=768, height=7, width=7),
            conv_dims=[],
            fc_dims=[1024, 1024],
        ),
        box_predictor=L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(10, 10, 5, 5)),
            num_classes="${..num_classes}",
        ),
        mask_in_features="${.box_in_features}",
        mask_pooler=L(ROIPooler)(
            output_size=14,
            scales=(1.0 / 16,),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        mask_head=L(MaskRCNNConvUpsampleHead)(
            input_shape=ShapeSpec(channels=768, width=14, height=14),
            num_classes="${..num_classes}",
            conv_dims=[256, 256, 256, 256, 256],
        ),
    ),
    pixel_mean=[123.675, 116.28, 103.53],
    pixel_std=[58.395, 57.12, 57.375],
    input_format="RGB",
)
