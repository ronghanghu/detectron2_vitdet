"""
This is an example config file.

The goal is to have as much flexibility as possible.
"""

import os

import torch

from torch_detectron.helpers.config import get_default_config
from torch_detectron.helpers.config_utils import import_file

config = get_default_config()

catalog = import_file(
    "torch_detectron.paths_catalog",
    os.path.join(os.path.dirname(__file__), "paths_catalog.py"),
)

# dataset

config.TRAIN.DATA.DATASET.FILES = [
    catalog.DatasetCatalog.get("coco_2014_train"),
    catalog.DatasetCatalog.get("coco_2014_valminusminival"),
]
config.TEST.DATA.DATASET.FILES = [catalog.DatasetCatalog.get("coco_2014_minival")]


# model
pretrained_path = catalog.ModelCatalog.get("R-50")
config.MODEL.BACKBONE.WEIGHTS = pretrained_path
config.MODEL.HEADS.WEIGHTS = pretrained_path
config.MODEL.RPN_ONLY = True

config.MODEL.RPN.PRE_NMS_TOP_N_TEST = 12000
config.MODEL.RPN.POST_NMS_TOP_N_TEST = 2000

num_gpus = 8

# training
images_per_batch = 2
lr = 0.00125 * num_gpus * images_per_batch

config.TRAIN.DATA.DATALOADER.IMAGES_PER_BATCH = images_per_batch

config.SOLVER.MAX_ITER = 90000
config.SOLVER.OPTIM.BASE_LR = lr
config.SOLVER.OPTIM.BASE_LR_BIAS = 2 * lr
config.SOLVER.OPTIM.WEIGHT_DECAY = 0.0001
config.SOLVER.OPTIM.WEIGHT_DECAY_BIAS = 0
config.SOLVER.OPTIM.MOMENTUM = 0.9

config.SOLVER.SCHEDULER.STEPS = [60000, 80000]
config.SOLVER.SCHEDULER.GAMMA = 0.1

config.SAVE_DIR = os.environ["SAVE_DIR"] if "SAVE_DIR" in os.environ else ""
config.CHECKPOINT = (
    os.environ["CHECKPOINT_FILE"] if "CHECKPOINT_FILE" in os.environ else ""
)

if "QUICK_SCHEDULE" in os.environ and os.environ["QUICK_SCHEDULE"]:
    config.TRAIN.DATA.DATASET.FILES = [catalog.DatasetCatalog.get("coco_2014_minival")]

    config.MODEL.RPN.PRE_NMS_TOP_N_TEST = 10000
    config.MODEL.RPN.POST_NMS_TOP_N_TEST = 2000

    lr = 0.01
    config.SOLVER.MAX_ITER = 2000
    config.SOLVER.OPTIM.BASE_LR = lr
    config.SOLVER.OPTIM.BASE_LR_BIAS = 2 * lr

    config.SOLVER.SCHEDULER.STEPS = [1000]
