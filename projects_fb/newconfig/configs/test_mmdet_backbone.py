# -*- coding: utf-8 -*-
from detectron2.config.instantiate import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.mmdet_wrapper import MMDetBackbone

from newconfig import ConfigFile

model, dataloader, lr_multiplier, optimizer, train = ConfigFile.load_rel(
    "./mask_rcnn_R_50_FPN_1x.py", ("model", "dataloader", "lr_multiplier", "optimizer", "train")
)

# preproc for torchvision models:
model.pixel_mean = [123.675, 116.280, 103.530]
model.pixel_std = [58.395, 57.120, 57.375]
dataloader.train.mapper.image_format = "RGB"

model.backbone = L(MMDetBackbone)(
    backbone=dict(
        type="DetectoRS_ResNet",
        conv_cfg=dict(type="ConvAWS"),
        sac=dict(type="SAC", use_deform=True),
        stage_with_sac=(False, True, True, True),
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
    ),
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
    ),
    pretrained_backbone="torchvision://resnet50",
    output_shapes=[ShapeSpec(channels=256, stride=s) for s in [4, 8, 16, 32, 64]],
    output_names=["p2", "p3", "p4", "p5", "p6"],
)

train.init_checkpoint = None
