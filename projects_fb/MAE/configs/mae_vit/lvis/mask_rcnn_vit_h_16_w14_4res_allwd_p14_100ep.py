from functools import partial
import torch.nn as nn
from fvcore.common.param_scheduler import MultiStepParamScheduler

import detectron2.data.transforms as T
from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.evaluation.lvis_evaluation import LVISEvaluator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads import FastRCNNOutputLayers
from detectron2.solver import WarmupParamScheduler
from detectron2.data.detection_utils import get_fed_loss_cls_weights

from ....vit.vit import ViTUpDimDown
from ....beit.beit import BEiTDet
from ...common.coco import dataloader
from ....beit.fpn import FPNWoTopdown

# Data using LSJ
image_size = 1024
dataloader.train.dataset.names = "lvis_v1_train"
dataloader.train.sampler = L(RepeatFactorTrainingSampler)(
    repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
        dataset_dicts="${dataloader.train.dataset}", repeat_thresh=0.001)
)
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

dataloader.test.dataset.names = "lvis_v1_val"
dataloader.evaluator = L(LVISEvaluator)(
    dataset_name="${..test.dataset.names}",
    max_dets_per_image=300,
)
dataloader.test.mapper.augmentations = [
    L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
    L(T.FixedSizeCrop)(crop_size=(image_size, image_size)),
]


# Model
# Base
# embed_dim, depth, num_heads = 768, 12, 12
# # Large
# embed_dim, depth, num_heads = 1024, 24, 16
# Huge
embed_dim, depth, num_heads = 1280, 32, 16

model = model_zoo.get_config("common/models/mask_rcnn_fpn.py").model
# To match the data loader
model.pixel_mean = [123.675, 116.28, 103.53]
model.pixel_std = [58.395, 57.12, 57.375]
model.input_format = "RGB"
model.backbone.bottom_up = L(ViTUpDimDown)(  # Creates multi-scale feature maps from ViT backbone
    net=L(BEiTDet)(  # Single-scale ViT backbone
        img_size=image_size,
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=0.4,  # 0.4 dp for ViT-H as mentioned in appendix Hyper-parameters for LVIS
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        checkpoint_block_num=100000,
        use_cls_token_det=False,
        use_shared_rel_pos_bias=False,
        init_values=None,
        # model size: L
        window_block_indexes=range(depth),
        residual_block="basic",
        residual_block_indexes=[7, 15, 23, 31],
        residual_norm="LN",
        residual_act="gelu",
        out_features=["s1", "s2", "s3", "s4"],
        out_block_indexes=[31, 31, 31, 31],
        rel_q=True,
    ),
    in_features="${.net.out_features}",
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    embed_dim="${.net.embed_dim}",
    mode=6,
)
model.backbone.in_features = "${.bottom_up.in_features}"
model.backbone.update(
    _target_=FPNWoTopdown,
)


# New heads and LN

model.backbone.norm = "LN"  # Use LN in FPN

model.roi_heads.box_head.conv_norm = (
    model.roi_heads.mask_head.conv_norm
) = "LN"

# 2conv in RPN:
# https://github.com/tensorflow/tpu/blob/b24729de804fdb751b06467d3dce0637fa652060/models/official/detection/modeling/architecture/heads.py#L95-L97  # noqa: E501, B950
model.proposal_generator.head.conv_dims = [-1, -1]

# 4conv1fc box head
model.roi_heads.box_head.conv_dims = [256, 256, 256, 256]
model.roi_heads.box_head.fc_dims = [1024]

# LVIS specific
model.roi_heads.num_classes = 1203

model.roi_heads.box_predictor = L(FastRCNNOutputLayers)(
    input_shape=ShapeSpec(channels=1024),
    test_score_thresh=0.02,
    box2box_transform=L(Box2BoxTransform)(weights=(10, 10, 5, 5)),
    num_classes="${..num_classes}",
    test_topk_per_image=300,
    use_sigmoid_ce=True,
    use_fed_loss=True,
    get_fed_loss_cls_weights=lambda: get_fed_loss_cls_weights(dataloader.train.dataset.names, 0.5),
)


# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = False
# from mae init (MAE-Large-removeMeanStd-1600ep, X% )
train.init_checkpoint = "/checkpoint/kaiminghe/converted/2021-10-27-01-02-32-v3-256-mb4096-epo1600-PMAEp14-ViTHuge-lr1e-4-wd5e-2-warm40-mask0.75-pred8d512-exNB-msaLNmlpLNeLNpLNkBN0-1view-NOrelpos-abspos-clstoken-qkv-NOlayerscale-LNtgteps1e-2/pretrained_lastnorm_tf2pt.pth"  
# from ...common.get_fid_dir import get_wd
# train.init_checkpoint = f"{get_wd('f333655988')}/model_final.pth"

# Schedule

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
# train.max_iter = train.max_iter // 2  # 100 ep -> 50 ep
# lr_multiplier.scheduler.milestones = [
#     milestone // 2 for milestone in lr_multiplier.scheduler.milestones
# ]
# lr_multiplier.scheduler.num_updates = train.max_iter

from ...common.optim import AdamLayerDecay as optimizer

# Optimized hyperparams
optimizer.lr = 1e-4  # 1e-4 lr is used for ViT-H on LVIS
optimizer.weight_decay = 0.1
optimizer.params.lr_decay_rate = 0.9
optimizer.params.num_layers = 32
optimizer.params.skip_lr_decay = ["residual."]


# wd for all params
optimizer.params.overrides = {}
optimizer.params.weight_decay_norm = None
