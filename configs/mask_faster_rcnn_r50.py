"""
This is an example config file.

The goal is to have as much flexibility as possible.

TODO: remove the dependency on get_data_loader,
get_model, get_optimizer, get_scheduler and the extra
arguments in the training code and use directly the config
"""

import os

import torch

from torch_detectron.helpers.config import get_default_config
from torch_detectron.helpers.config_utils import ConfigClass
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
config.TEST.DATA.DATASET.FILES = [
    catalog.DatasetCatalog.get("coco_2014_minival"),
]


def head_builder(pretrained_path):
    from torch_detectron.model_builder.resnet import Bottleneck, ResNetHead

    block = Bottleneck
    # head = ResNetHead(block, layers=[(5, 3)], stride_init=1)
    head = ResNetHead(block, layers=[(5, 3)])
    if pretrained_path:
        state_dict = torch.load(pretrained_path)
        head.load_state_dict(state_dict, strict=False)
    return head


def classifier(num_classes, pretrained_path=None):
    from torch_detectron.model_builder.resnet import Bottleneck, ClassifierHead

    classifier = ClassifierHead(512 * Bottleneck.expansion, num_classes)
    if pretrained_path:
        state_dict = torch.load(pretrained_path)
        classifier.load_state_dict(state_dict, strict=False)
    return classifier


def mask_classifier(num_classes, pretrained_path=None):
    from torch import nn

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


# model
pretrained_path = catalog.ModelCatalog.get('R-50')
config.MODEL.BACKBONE.WEIGHTS = pretrained_path
config.MODEL.HEADS.WEIGHTS = pretrained_path
config.MODEL.HEADS.BUILDER = head_builder
config.MODEL.HEADS.HEAD_BUILDER = classifier
config.MODEL.HEADS.MASK_BUILDER = mask_classifier

config.MODEL.USE_MASK = True


config.MODEL.REGION_PROPOSAL.PRE_NMS_TOP_N = 12000
config.MODEL.REGION_PROPOSAL.POST_NMS_TOP_N = 2000

config.MODEL.REGION_PROPOSAL.PRE_NMS_TOP_N_TEST = 6000
config.MODEL.REGION_PROPOSAL.POST_NMS_TOP_N_TEST = 1000

if False:
    lr = 0.005
    config.MODEL.HEADS.BATCH_SIZE_PER_IMAGE = 256

    import torch_detectron.layers

    config.MODEL.HEADS.POOLER.MODULE = torch_detectron.layers.ROIAlign(
        (7, 7), 1.0 / 16, 0
    )
    # import os
    # os.environ['SAVE_DIR'] = '/checkpoint02/fmassa/detectron_logs/mask_rcnn_quick_debug'

num_gpus = 8

# training
images_per_batch = 1  # 1
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


config.SAVE_DIR = os.environ["SAVE_DIR"] if "SAVE_DIR" in os.environ else ""
config.CHECKPOINT = (
    os.environ["CHECKPOINT_FILE"] if "CHECKPOINT_FILE" in os.environ else ""
)
