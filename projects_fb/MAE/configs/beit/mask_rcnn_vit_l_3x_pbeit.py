from fvcore.common.param_scheduler import MultiStepParamScheduler
import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from functools import partial
import torch.nn as nn

from detectron2 import model_zoo
from detectron2.layers import ShapeSpec


train = model_zoo.get_config("common/train.py").train

from ..common.coco import dataloader
from ..common.optim import AdamW as optimizer
from ..common.vit.vit import model

from ...beit.beit import BEiTDet

# LSJ
image_size = 1024
dataloader.train.mapper.augmentations = [
    L(T.ResizeScale)(
        min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
    ),
    L(T.FixedSizeCrop)(crop_size=(image_size, image_size)),
    L(T.RandomFlip)(horizontal=True),
]

# recompute boxes due to cropping
dataloader.train.mapper.recompute_boxes = True

dataloader.test.mapper.augmentations = [
    L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
    L(T.FixedSizeCrop)(crop_size=(image_size, image_size)),
]


train.amp.enabled = True
train.ddp.fp16_compression = False
train.init_checkpoint = "/checkpoint/kaiminghe/logs/deit/210820-123904-OrgIN1K-8n-BeiT-Large-ep800-lr1.5e-3-logred-abspos/checkpoint-799.pth"

num_node = 8

dataloader.train.total_batch_size = 8 * num_node
optimizer.lr = 0.00002 * num_node

# 3x
total_steps = 3 * 90000 * 2  # bs = 8
train.max_iter = int(total_steps / num_node)

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[
            int((total_steps - 120000) / num_node),
            int((total_steps - 40000) / num_node),
            int(total_steps / num_node),
        ],
    ),
    warmup_length=2000 / num_node / train.max_iter,
    warmup_factor=0.001,
)

model.backbone = L(BEiTDet)(
    img_size=image_size,
    embed_dim=1024,
    depth=24,
    num_heads=16,
    mlp_ratio=4,
    qkv_bias=True,
    init_values=1e-5,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    out_features=["block23"],
    checkpoint_block_num=20,
)
model.roi_heads.box_head.input_shape = ShapeSpec(channels=1024, height=7, width=7)
model.roi_heads.mask_head.input_shape = ShapeSpec(channels=1024, height=14, width=14)
