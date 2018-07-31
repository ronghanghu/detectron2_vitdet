"""
This is an example config file.

The goal is to have as much flexibility as possible.
"""

import os

import torch
from torch import nn

from torch_detectron.helpers.config import get_default_config
from torch_detectron.helpers.config import set_resnet_defaults
from torch_detectron.helpers.config import set_roi_heads_defaults
from torch_detectron.helpers.config import set_rpn_defaults
from torch_detectron.helpers.config_utils import import_file
from torch_detectron.model_builder import resnet

catalog = import_file(
    "torch_detectron.paths_catalog",
    os.path.join(os.path.dirname(__file__), "paths_catalog.py"),
)
pretrained_path = catalog.ModelCatalog.get("R-50")


# --------------------------------------------------------------------------------------
# Default config options
# --------------------------------------------------------------------------------------
config = get_default_config()
set_resnet_defaults(config)
set_rpn_defaults(config)
set_roi_heads_defaults(config)


# --------------------------------------------------------------------------------------
# Training and testing data
# --------------------------------------------------------------------------------------
config.TRAIN.DATA.DATASET.FILES = [
    catalog.DatasetCatalog.get("coco_2014_train"),
    catalog.DatasetCatalog.get("coco_2014_valminusminival"),
]
config.TEST.DATA.DATASET.FILES = [catalog.DatasetCatalog.get("coco_2014_minival")]
config.TRAIN.DATA.DATALOADER.BATCH_SAMPLER.IMAGES_PER_BATCH = 1


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
config.MODEL.USE_MASK = True


# --------------------------------------------------------------------------------------
# Backbone
# --------------------------------------------------------------------------------------
config.MODEL.BACKBONE.WEIGHTS = pretrained_path


# --------------------------------------------------------------------------------------
# RPN
# --------------------------------------------------------------------------------------
config.MODEL.RPN.PRE_NMS_TOP_N = 12000
config.MODEL.RPN.POST_NMS_TOP_N = 2000
config.MODEL.RPN.PRE_NMS_TOP_N_TEST = 6000
config.MODEL.RPN.POST_NMS_TOP_N_TEST = 1000


# --------------------------------------------------------------------------------------
# RoI heads
# --------------------------------------------------------------------------------------
def head_builder(config, pretrained_path):
    stage = resnet.StageSpec(index=5, block_count=3, return_features=False)
    head = resnet.ResNetHead(
        block_module=config.MODEL.RESNET.BLOCK_MODULE,
        stages=(stage,),
        num_groups=config.MODEL.RESNET.NUM_GROUPS,
        width_per_group=config.MODEL.RESNET.WIDTH_PER_GROUP,
        stride_in_1x1=config.MODEL.RESNET.STRIDE_IN_1X1,
    )
    if pretrained_path:
        state_dict = torch.load(pretrained_path)
        head.load_state_dict(state_dict, strict=False)
    return head


def classifier(config, num_classes, pretrained_path=None):
    classifier = resnet.ClassifierHead(2048, num_classes)
    if pretrained_path:
        state_dict = torch.load(pretrained_path)
        classifier.load_state_dict(state_dict, strict=False)
    return classifier


def mask_classifier(config, num_classes, pretrained_path=None):
    model = nn.Sequential(
        nn.ConvTranspose2d(512 * 4, 256, 2, 2, 0),
        nn.ReLU(),
        nn.Conv2d(256, num_classes, 1, 1, 0),
    )
    for name, param in model.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    if pretrained_path:
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict, strict=False)
    return model


config.MODEL.ROI_HEADS.WEIGHTS = pretrained_path
config.MODEL.ROI_HEADS.BUILDER = head_builder
config.MODEL.ROI_HEADS.HEAD_BUILDER = classifier
config.MODEL.ROI_HEADS.MASK_BUILDER = mask_classifier


# --------------------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------------------
num_gpus = 8
lr = 0.00125 * num_gpus * config.TRAIN.DATA.DATALOADER.BATCH_SAMPLER.IMAGES_PER_BATCH
config.SOLVER.OPTIM.BASE_LR = lr
config.SOLVER.OPTIM.BASE_LR_BIAS = 2 * lr
config.SOLVER.OPTIM.WEIGHT_DECAY = 0.0001
config.SOLVER.MAX_ITER = 180000
config.SOLVER.SCHEDULER.STEPS = [120000, 160000]

config.SAVE_DIR = os.environ["SAVE_DIR"] if "SAVE_DIR" in os.environ else ""
config.CHECKPOINT = (
    os.environ["CHECKPOINT_FILE"] if "CHECKPOINT_FILE" in os.environ else ""
)


# --------------------------------------------------------------------------------------
# Quick config
# --------------------------------------------------------------------------------------
if "QUICK_SCHEDULE" in os.environ and os.environ["QUICK_SCHEDULE"]:
    config.TRAIN.DATA.DATASET.FILES = [catalog.DatasetCatalog.get("coco_2014_minival")]

    config.TRAIN.DATA.TRANSFORM.MIN_SIZE = 600
    config.TRAIN.DATA.TRANSFORM.MAX_SIZE = 1000
    config.TEST.DATA.TRANSFORM.MIN_SIZE = 800
    config.TEST.DATA.TRANSFORM.MAX_SIZE = 1000

    config.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256

    lr = 0.005
    config.SOLVER.MAX_ITER = 2000
    config.SOLVER.OPTIM.BASE_LR = lr
    config.SOLVER.OPTIM.BASE_LR_BIAS = 2 * lr

    config.SOLVER.SCHEDULER.STEPS = [1500]

    config.TEST.EXPECTED_RESULTS: [['bbox', 'AP', [0.035681, 0.000825]], ['segm', 'AP', [0.043628, 0.000940]]]
