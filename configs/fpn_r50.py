"""
This is an example config file.

The goal is to have as much flexibility as possible.

TODO: remove the dependency on get_data_loader,
get_model, get_optimizer, get_scheduler and the extra
arguments in the training code and use directly the config
"""

import torch
from torch_detectron.helpers.config import config
from torch_detectron.helpers.config_utils import ConfigClass

from torch_detectron.core.fpn import fpn_resnet50_conv5_body, FPNPooler, fpn_classification_head

# dataset
config.TRAIN.DATA.DATASET.FILES = [
        ('/datasets01/COCO/060817/annotations/instances_train2014.json',
         '/datasets01/COCO/060817/train2014/'),
        ('/private/home/fmassa/coco_trainval2017/annotations/instances_valminusminival2017.json',
         '/datasets01/COCO/060817/val2014/')
]
config.TEST.DATA.DATASET.FILES = [
        ('/private/home/fmassa/coco_trainval2017/annotations/instances_val2017_mod.json',
         '/datasets01/COCO/060817/val2014/')
]


config.TRAIN.DATA.DATALOADER.COLLATOR.SIZE_DIVISIBLE = 32
config.TEST.DATA.DATALOADER.COLLATOR.SIZE_DIVISIBLE = 32

# model

class Pooler(ConfigClass):
    def __call__(self):
        return FPNPooler(
            output_size=(7, 7), scales=[2 ** (-i) for i in range(2, 6)], sampling_ratio=2, drop_last=True)

pretrained_path = '/private/home/fmassa/github/detectron.pytorch/torch_detectron/core/models/r50_new.pth'
# pretrained_path = '/private/home/fmassa/github/detectron.pytorch/torch_detectron/core/models/fpn_r50.pth'

config.MODEL.BACKBONE.WEIGHTS = pretrained_path
config.MODEL.HEADS.WEIGHTS = pretrained_path

config.MODEL.BACKBONE.BUILDER = fpn_resnet50_conv5_body
config.MODEL.HEADS.BUILDER = fpn_classification_head
config.MODEL.HEADS.USE_FPN = True
config.MODEL.HEADS.POOLER = Pooler()
config.MODEL.REGION_PROPOSAL.USE_FPN = True
config.MODEL.REGION_PROPOSAL.NUM_INPUT_FEATURES = 256
config.MODEL.REGION_PROPOSAL.ANCHOR_STRIDE = (4, 8, 16, 32, 64)
config.MODEL.REGION_PROPOSAL.PRE_NMS_TOP_N = 2000

config.MODEL.REGION_PROPOSAL.PRE_NMS_TOP_N_TEST = 1000
config.MODEL.REGION_PROPOSAL.POST_NMS_TOP_N_TEST = 1000
config.MODEL.REGION_PROPOSAL.FPN_POST_NMS_TOP_N_TEST = 1000


# only for quick

if False:
    config.MODEL.REGION_PROPOSAL.PRE_NMS_TOP_N = 12000
    config.MODEL.REGION_PROPOSAL.POST_NMS_TOP_N = 2000
    config.MODEL.REGION_PROPOSAL.POSITIVE_FRACTION = 0.25
    config.MODEL.REGION_PROPOSAL.STRADDLE_THRESH = -1
    config.MODEL.REGION_PROPOSAL.PRE_NMS_TOP_N_TEST = 1000
    config.MODEL.REGION_PROPOSAL.POST_NMS_TOP_N_TEST = 2000
    config.TRAIN.DATA.TRANSFORM.MIN_SIZE = 600
    config.TRAIN.DATA.TRANSFORM.MAX_SIZE = 1000
    config.TEST.DATA.TRANSFORM.MIN_SIZE = 600
    config.TEST.DATA.TRANSFORM.MAX_SIZE = 1000

    loaded_weights = torch.load(pretrained_path)
    from collections import OrderedDict
    rpn_weights = OrderedDict()
    for k, v in loaded_weights.items():
        if k.startswith('rpn'):
            rpn_weights[k[4:]] = v

    config.MODEL.REGION_PROPOSAL.WEIGHTS = rpn_weights


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

import os
config.SAVE_DIR = os.environ['SAVE_DIR'] if 'SAVE_DIR' in os.environ else ''
config.CHECKPOINT = os.environ['CHECKPOINT_FILE'] if 'CHECKPOINT_FILE' in os.environ else ''
