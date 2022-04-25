from functools import partial
import torch.nn as nn
from fvcore.common.param_scheduler import MultiStepParamScheduler

import detectron2.data.transforms as T
from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.layers.batch_norm import NaiveSyncBatchNorm
from detectron2.solver import WarmupParamScheduler

from ....vit.vit import ViTUp1
from ....beit.beit import BEiTDet
from ...common.coco import dataloader
from ...common.optim import AdamW as optimizer

# Data using LSJ
image_size = 896 # 896 / 14 = 64 (stride=14)
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


# Model
# Base
# embed_dim, depth, num_heads = 768, 12, 12
# Large
# embed_dim, depth, num_heads = 1024, 24, 16
# Huge
embed_dim, depth, num_heads = 1280, 32, 16

model = model_zoo.get_config("common/models/mask_rcnn_fpn.py").model
# To match the data loader
model.pixel_mean = [123.675, 116.28, 103.53]
model.pixel_std = [58.395, 57.12, 57.375]
model.input_format = "RGB"
model.backbone.bottom_up = L(ViTUp1)(  # Creates multi-scale feature maps from ViT backbone
    net=L(BEiTDet)(  # Single-scale ViT backbone
        img_size=image_size,
        patch_size=14,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=0.0,
        window_size=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        checkpoint_block_num=32,
        use_cls_token_det=False,
        use_shared_rel_pos_bias=True,
        init_values=None, 
        # model size: H
        window_block_indexes=[
            0, 1, 2, 3, 4, 5, 6,         # 7 global
            8, 9, 10, 11, 12, 13, 14,    # 15 global
            16, 17, 18, 19, 20, 21, 22,  # 23 global
            24, 25, 26, 27, 28, 29, 30,  # 31 global
        ],
        out_features=["block7", "block15", "block23", "block31"],
    ),
    in_features="${.net.out_features}",
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    embed_dim="${.net.embed_dim}",
    mode=5,
    resize_ratio=14.0 / 16,   # Resize features with stride=16 for ViT-H
)
model.backbone.in_features = "${.bottom_up.net.out_features}"


# New heads and syncbn

model.backbone.norm = "SyncBN"  # Use SyncBN in FPN

# Using NaiveSyncBatchNorm becase heads may have empty input. That is not supported by
# torch.nn.SyncBatchNorm. We can remove this after
# https://github.com/pytorch/pytorch/issues/36530 is fixed.
model.roi_heads.box_head.conv_norm = (
    model.roi_heads.mask_head.conv_norm
) = lambda c: NaiveSyncBatchNorm(c, stats_mode="N")

# 2conv in RPN:
# https://github.com/tensorflow/tpu/blob/b24729de804fdb751b06467d3dce0637fa652060/models/official/detection/modeling/architecture/heads.py#L95-L97  # noqa: E501, B950
model.proposal_generator.head.conv_dims = [-1, -1]

# 4conv1fc box head
model.roi_heads.box_head.conv_dims = [256, 256, 256, 256]
model.roi_heads.box_head.fc_dims = [1024]


# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = False
# from mae init (MAE-Huge-vqvaeIN1k-1600ep, X% )
train.init_checkpoint = "/checkpoint/kaiminghe/converted/2021-10-27-01-02-32-v3-256-mb4096-epo1600-PMAEp14-ViTHuge-lr1e-4-wd5e-2-warm40-mask0.75-pred8d512-exNB-msaLNmlpLNeLNpLNkBN0-1view-NOrelpos-abspos-clstoken-qkv-NOlayerscale-LNtgteps1e-2/pretrained_lastnorm_tf2pt.pth"  




# Schedule

# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
train.max_iter = 184375
num_node = 8

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[163889, 177546],
        num_updates=train.max_iter,
    ),
    warmup_length=2000 / num_node / train.max_iter,
    warmup_factor=0.001,
)

# Rescale schedule
train.max_iter = train.max_iter // 8  # 100ep -> 12.5ep
lr_multiplier.scheduler.milestones = [
    milestone // 8 for milestone in lr_multiplier.scheduler.milestones
]
lr_multiplier.scheduler.num_updates = train.max_iter


# Optimizer hyperparams
# center = (1.6e-4, 0.1)
optimizer.lr = 1.6e-4
optimizer.weight_decay = 0.1
optimizer.params.overrides = {
    "pos_embed": {"weight_decay": 0.0},
    "relative_position_bias_table": {"weight_decay": 0.0},
}
