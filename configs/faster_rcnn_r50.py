"""
This is an example config file.

The goal is to have as much flexibility as possible.

TODO: remove the dependency on get_data_loader,
get_model, get_optimizer, get_scheduler and the extra
arguments in the training code and use directly the config
"""

import torch
from torch_detectron.helpers.config import config


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

num_gpus = 8

# training
images_per_batch = 1
lr = 0.00125 * num_gpus * images_per_batch

config.TRAIN.DATA.DATALOADER.IMAGES_PER_BATCH = images_per_batch

config.SOLVER.MAX_ITER = 180000
config.SOLVER.OPTIM.BASE_LR = lr
config.SOLVER.OPTIM.BASE_LR_BIAS = 2 * lr
config.SOLVER.OPTIM.WEIGHT_DECAY = 0.0001
config.SOLVER.OPTIM.WEIGHT_DECAY_BIAS = 0
config.SOLVER.OPTIM.MOMENTUM = 0.9

config.SOLVER.SCHEDULER.STEPS = [120000, 160000]
config.SOLVER.SCHEDULER.GAMMA = 0.1


# FIXME remove those required attributes from the trainer and directly use the config
# instead
device = config.DEVICE
max_iter = config.SOLVER.MAX_ITER

import os
save_dir = os.environ['SAVE_DIR'] if 'SAVE_DIR' in os.environ else ''
checkpoint = os.environ['CHECKPOINT_FILE'] if 'CHECKPOINT_FILE' in os.environ else ''
do_test = config.DO_TEST

# function getters
# should implement get_data_loader, get_model, get_optimizer and get_scheduler
# FIXME remove there and use directly the config
def get_data_loader(distributed=False):
    config.TRAIN.DATA.DATALOADER.SAMPLER.DISTRIBUTED = distributed
    config.TEST.DATA.DATALOADER.SAMPLER.DISTRIBUTED = distributed

    data_loader = config.TRAIN.DATA()
    data_loader_val = config.TEST.DATA()
    return data_loader, data_loader_val


def get_model():
    model = config.MODEL()
    model.to(device)
    return model

get_optimizer = config.SOLVER.OPTIM
get_scheduler = config.SOLVER.SCHEDULER
