"""
This is an example config file.

The goal is to have as much flexibility as possible.
"""

import os

import torch

from torch_detectron.helpers.config import get_default_config

config = get_default_config()

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


# model
pretrained_path = '/private/home/fmassa/github/detectron.pytorch/torch_detectron/core/models/r50.pth'
config.MODEL.BACKBONE.WEIGHTS = pretrained_path
config.MODEL.HEADS.WEIGHTS = pretrained_path
config.MODEL.RPN_ONLY = True

config.MODEL.REGION_PROPOSAL.PRE_NMS_TOP_N_TEST = 12000
config.MODEL.REGION_PROPOSAL.POST_NMS_TOP_N_TEST = 2000

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

config.SAVE_DIR = os.environ['SAVE_DIR'] if 'SAVE_DIR' in os.environ else ''
config.CHECKPOINT = os.environ['CHECKPOINT_FILE'] if 'CHECKPOINT_FILE' in os.environ else ''

if 'QUICK_SCHEDULE' in os.environ and os.environ['QUICK_SCHEDULE']:
    config.TRAIN.DATA.DATASET.FILES = [
            ('/private/home/fmassa/coco_trainval2017/annotations/instances_val2017_mod.json',
             '/datasets01/COCO/060817/val2014/')
    ]


    config.MODEL.REGION_PROPOSAL.PRE_NMS_TOP_N_TEST = 10000
    config.MODEL.REGION_PROPOSAL.POST_NMS_TOP_N_TEST = 2000

    lr = 0.01
    config.SOLVER.MAX_ITER = 2000
    config.SOLVER.OPTIM.BASE_LR = lr
    config.SOLVER.OPTIM.BASE_LR_BIAS = 2 * lr

    config.SOLVER.SCHEDULER.STEPS = [1000]
