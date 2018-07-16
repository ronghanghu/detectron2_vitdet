"""
This is an example config file.

The goal is to have as much flexibility as possible.
"""

import os

from torch_detectron.core.box_selector import ROI2FPNLevelsMapper
from torch_detectron.core.fpn import FPNPooler
from torch_detectron.core.mask_rcnn import MaskFPNPooler
from torch_detectron.helpers.config import get_default_config
from torch_detectron.helpers.config import set_rpn_defaults
from torch_detectron.helpers.config import set_roi_heads_defaults
from torch_detectron.helpers.config_utils import ConfigNode
from torch_detectron.helpers.config_utils import import_file
from torch_detectron.helpers.model import fpn_classification_head
from torch_detectron.helpers.model import fpn_resnet50_conv5_body
from torch_detectron.helpers.model import maskrcnn_head

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
config.TRAIN.DATA.DATALOADER.COLLATOR.SIZE_DIVISIBLE = 32
config.TEST.DATA.DATALOADER.COLLATOR.SIZE_DIVISIBLE = 32
config.TRAIN.DATA.DATALOADER.BATCH_SAMPLER.IMAGES_PER_BATCH = 2


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
config.MODEL.USE_MASK = True


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
config.MODEL.RPN.PRE_NMS_TOP_N = 2000
config.MODEL.RPN.POST_NMS_TOP_N = 2000
config.MODEL.RPN.PRE_NMS_TOP_N_TEST = 1000
config.MODEL.RPN.POST_NMS_TOP_N_TEST = 1000
config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 1000


# --------------------------------------------------------------------------------------
# RoI heads
# --------------------------------------------------------------------------------------
class Pooler(ConfigNode):
    def __call__(self):
        return FPNPooler(
            output_size=(7, 7),
            scales=[2 ** (-i) for i in range(2, 6)],
            sampling_ratio=2,
            drop_last=True,
        )


class MaskPooler(ConfigNode):
    def __call__(self):
        roi_to_fpn_level_mapper = ROI2FPNLevelsMapper(2, 5)
        return MaskFPNPooler(
            roi_to_fpn_level_mapper=roi_to_fpn_level_mapper,
            output_size=(14, 14),
            scales=[2 ** (-i) for i in range(2, 6)],
            sampling_ratio=2,
            drop_last=True,
        )


config.MODEL.ROI_HEADS.BUILDER = fpn_classification_head
config.MODEL.ROI_HEADS.USE_FPN = True
config.MODEL.ROI_HEADS.POOLER = Pooler()
config.MODEL.ROI_HEADS.MASK_POOLER = MaskPooler()
config.MODEL.ROI_HEADS.MASK_RESOLUTION = 28
config.MODEL.ROI_HEADS.MASK_BUILDER = maskrcnn_head


# --------------------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------------------
num_gpus = 8
lr = 0.00125 * num_gpus * config.TRAIN.DATA.DATALOADER.BATCH_SAMPLER.IMAGES_PER_BATCH
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
# Quick config
# --------------------------------------------------------------------------------------
if "QUICK_SCHEDULE" in os.environ and os.environ["QUICK_SCHEDULE"]:
    raise NotImplementedError("Not fully implemented")
    config.TRAIN.DATA.DATASET.FILES = [catalog.DatasetCatalog.get("coco_2014_minival")]
    lr = 0.005
    config.SOLVER.OPTIM.BASE_LR = lr
    config.SOLVER.OPTIM.BASE_LR_BIAS = 2 * lr

    config.SOLVER.MAX_ITER = 2000
    config.SOLVER.SCHEDULER.STEPS = [1000]
