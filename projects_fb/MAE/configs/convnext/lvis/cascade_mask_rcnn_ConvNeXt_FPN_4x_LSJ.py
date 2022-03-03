from detectron2 import model_zoo
from detectron2.config.lazy import LazyCall as L
import detectron2.data.transforms as T
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import FastRCNNOutputLayers, FastRCNNConvFCHead, CascadeROIHeads
from detectron2.solver import WarmupParamScheduler
from detectron2.solver.build import get_default_optimizer_params
from detectron2.modeling.backbone.convnext import ConvNeXt
from fvcore.common.param_scheduler import MultiStepParamScheduler

from ...common.lvis import dataloader
from ...common.optim import AdamW as optimizer

size2config = {
    "B": {
        "depths": [3, 3, 27, 3],
        "dims": [128, 256, 512, 1024], 
        "drop_path_rate": 0.7,
        "pretrained": "/checkpoint/hannamao/convnext/convnext_base_1k_224.pth",
        "lr": 0.0002,
    },
    "B-22k": {
        "depths": [3, 3, 27, 3],
        "dims": [128, 256, 512, 1024], 
        "drop_path_rate": 0.6,
        "pretrained": "/checkpoint/hannamao/convnext/convnext_base_22k_224.pth",
        "lr": 0.0001,
    },
    "L-22k": {
        "depths": [3, 3, 27, 3],
        "dims": [192, 384, 768, 1536], 
        "drop_path_rate": 0.7,
        "pretrained": "/checkpoint/hannamao/convnext/convnext_large_22k_224.pth",
        "lr": 0.0001,
    },
}

config = size2config["L-22k"]

train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = False
train.init_checkpoint = config["pretrained"]

model = model_zoo.get_config("common/models/mask_rcnn_fpn.py").model
model.pixel_mean = [123.675, 116.28, 103.53]
model.pixel_std = [58.395, 57.12, 57.375]
model.input_format = "RGB"

model.backbone.in_features = ["convnext0", "convnext1", "convnext2", "convnext3"]
model.backbone.bottom_up = L(ConvNeXt)(
    depths=config["depths"],
    dims=config["dims"],
    drop_path_rate=config["drop_path_rate"],
    layer_scale_init_value=1.0,
    out_indices=[0, 1, 2, 3],
)

# New heads and LN
model.backbone.norm = "LN"  # Use LN in FPN

model.roi_heads.box_head.conv_norm = (
    model.roi_heads.mask_head.conv_norm
) = "LN"

# 2conv in RPN:
# https://github.com/tensorflow/tpu/blob/b24729de804fdb751b06467d3dce0637fa652060/models/official/detection/modeling/architecture/heads.py#L95-L97  # noqa: E501, B950
model.proposal_generator.head.conv_dims = [-1, -1]

# LVIS specific   
model.roi_heads.num_classes = 1203

[model.roi_heads.pop(k) for k in ["box_head", "box_predictor", "proposal_matcher"]]

model.roi_heads.update(
    _target_=CascadeROIHeads,
    num_classes=1203,
    box_heads=[
        L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[256, 256, 256, 256],
            fc_dims=[1024],
            conv_norm="LN",
        )
        for _ in range(3)
    ],
    box_predictors=[
        L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            box2box_transform=L(Box2BoxTransform)(weights=(w1, w1, w2, w2)),
            num_classes="${...num_classes}", 
            test_score_thresh=0.02,
            test_topk_per_image=300,
            cls_agnostic_bbox_reg=True,
            use_sigmoid_ce=True,
            use_fed_loss=True,
        )
        for (w1, w2) in [(10, 5), (20, 10), (30, 15)]
    ],
    proposal_matchers=[
        L(Matcher)(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
        for th in [0.5, 0.6, 0.7]
    ],
)

model.roi_heads.mask_head.num_classes = 1  # using class agnostic prediction for mask

# Data using LSJ
image_size = 1024
dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(horizontal=True),  # flip first
    L(T.ResizeScale)(
        min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
    ),
    L(T.FixedSizeCrop)(crop_size=(image_size, image_size)),
]
dataloader.train.total_batch_size = 64
# recompute boxes due to cropping
dataloader.train.mapper.recompute_boxes = True

dataloader.test.mapper.augmentations = [
    L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
    L(T.FixedSizeCrop)(crop_size=(image_size, image_size)),
]

# 100 ep = 156250 iters * 64 images/iter / 100000 images/ep
train.max_iter = 156250
train.eval_period = 78125
num_node = 8

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[138889, 150463],  # following training schedules in table 15 of https://arxiv.org/abs/2101.11605v1
        num_updates=train.max_iter,
    ),
    warmup_length=2000 / num_node / train.max_iter,
    warmup_factor=0.001,
)

# Rescale schedule
train.max_iter = train.max_iter // 2  # 100 ep -> 50 ep
lr_multiplier.scheduler.milestones = [
    milestone // 2 for milestone in lr_multiplier.scheduler.milestones
]
lr_multiplier.scheduler.num_updates = train.max_iter

# Optimized hyperparams
optimizer.params=L(get_default_optimizer_params)(weight_decay_norm=0.0)
optimizer.lr = config["lr"]
optimizer.weight_decay = 0.05

