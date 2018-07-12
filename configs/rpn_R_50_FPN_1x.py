"""
This is an example config file.

The goal is to have as much flexibility as possible.
"""

import os

import torch

from torch_detectron.helpers.config import get_default_config
from torch_detectron.helpers.config import set_rpn_defaults
from torch_detectron.helpers.config_utils import import_file
from torch_detectron.helpers.model import fpn_resnet50_conv5_body

catalog = import_file(
    "torch_detectron.paths_catalog",
    os.path.join(os.path.dirname(__file__), "paths_catalog.py"),
)
pretrained_path = catalog.ModelCatalog.get("R-50")


# --------------------------------------------------------------------------------------
# Default config options
# --------------------------------------------------------------------------------------
config = get_default_config()
set_rpn_defaults(config)


# --------------------------------------------------------------------------------------
# Training and testing data
# --------------------------------------------------------------------------------------
config.TRAIN.DATA.DATASET.FILES = [
    catalog.DatasetCatalog.get("coco_2014_train"),
    catalog.DatasetCatalog.get("coco_2014_valminusminival"),
]
config.TEST.DATA.DATASET.FILES = [catalog.DatasetCatalog.get("coco_2014_minival")]
config.TRAIN.DATA.DATALOADER.COLLATOR.SIZE_DIVISIBLE = 32
config.TEST.DATA.DATALOADER.COLLATOR.SIZE_DIVISIBLE = 32
config.TRAIN.DATA.DATALOADER.IMAGES_PER_BATCH = 2


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
config.MODEL.RPN_ONLY = True


# --------------------------------------------------------------------------------------
# Backbone
# --------------------------------------------------------------------------------------
config.MODEL.BACKBONE.WEIGHTS = pretrained_path
config.MODEL.BACKBONE.BUILDER = fpn_resnet50_conv5_body
config.MODEL.BACKBONE.OUTPUT_DIM = 256


# --------------------------------------------------------------------------------------
# RPN
# --------------------------------------------------------------------------------------
config.MODEL.RPN.USE_FPN = True
config.MODEL.RPN.ANCHOR_STRIDE = (4, 8, 16, 32, 64)
config.MODEL.RPN.PRE_NMS_TOP_N_TEST = 1000
config.MODEL.RPN.POST_NMS_TOP_N_TEST = 2000


# --------------------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------------------
num_gpus = 8
lr = 0.00125 * num_gpus * config.TRAIN.DATA.DATALOADER.IMAGES_PER_BATCH
config.SOLVER.OPTIM.BASE_LR = lr
config.SOLVER.OPTIM.BASE_LR_BIAS = 2 * lr
config.SOLVER.OPTIM.WEIGHT_DECAY = 0.0001
config.SOLVER.MAX_ITER = 90000
config.SOLVER.SCHEDULER.STEPS = [60000, 80000]


config.SAVE_DIR = os.environ["SAVE_DIR"] if "SAVE_DIR" in os.environ else ""
config.CHECKPOINT = (
    os.environ["CHECKPOINT_FILE"] if "CHECKPOINT_FILE" in os.environ else ""
)


# --------------------------------------------------------------------------------------
# Quck config
# --------------------------------------------------------------------------------------
if "QUICK_SCHEDULE" in os.environ and os.environ["QUICK_SCHEDULE"]:
    config.TRAIN.DATA.DATASET.FILES = [catalog.DatasetCatalog.get("coco_2014_minival")]

    config.MODEL.RPN.PRE_NMS_TOP_N = 12000
    config.MODEL.RPN.POST_NMS_TOP_N = 2000
    config.MODEL.RPN.POSITIVE_FRACTION = 0.25
    config.MODEL.RPN.STRADDLE_THRESH = -1
    config.MODEL.RPN.PRE_NMS_TOP_N_TEST = 1000
    config.MODEL.RPN.POST_NMS_TOP_N_TEST = 2000
    config.TRAIN.DATA.TRANSFORM.MIN_SIZE = 600
    config.TRAIN.DATA.TRANSFORM.MAX_SIZE = 1000
    config.TEST.DATA.TRANSFORM.MIN_SIZE = 600
    config.TEST.DATA.TRANSFORM.MAX_SIZE = 1000

    lr = 0.005
    config.SOLVER.MAX_ITER = 2000
    config.SOLVER.OPTIM.BASE_LR = lr
    config.SOLVER.OPTIM.BASE_LR_BIAS = 2 * lr
    config.SOLVER.SCHEDULER.STEPS = [1000]

    _loaded_weights = torch.load(pretrained_path)
    from collections import OrderedDict

    _rpn_weights = OrderedDict()
    for k, _v in _loaded_weights.items():
        if k.startswith("rpn"):
            _rpn_weights[k[4:]] = _v

    config.MODEL.RPN.WEIGHTS = _rpn_weights
