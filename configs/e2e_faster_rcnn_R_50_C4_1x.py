"""
This is an example config file.

The goal is to have as much flexibility as possible.
"""

import os

from torch_detectron.helpers.config import get_default_config
from torch_detectron.helpers.config import set_rpn_defaults
from torch_detectron.helpers.config import set_roi_heads_defaults
from torch_detectron.helpers.config_utils import import_file

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
# Backbone
# --------------------------------------------------------------------------------------
config.MODEL.BACKBONE.WEIGHTS = pretrained_path


# --------------------------------------------------------------------------------------
# RoI heads
# --------------------------------------------------------------------------------------
config.MODEL.ROI_HEADS.WEIGHTS = pretrained_path


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

    config.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    config.TRAIN.DATA.DATALOADER.BATCH_SAMPLER.IMAGES_PER_BATCH = 2

    lr = 0.005
    config.SOLVER.MAX_ITER = 2000
    config.SOLVER.OPTIM.BASE_LR = lr
    config.SOLVER.OPTIM.BASE_LR_BIAS = 2 * lr

    config.SOLVER.SCHEDULER.STEPS = [1000]
