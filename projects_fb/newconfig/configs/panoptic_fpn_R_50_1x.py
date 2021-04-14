from detectron2.model_zoo import get_config

from .common_train import train

optimizer = get_config("common/common_optim.py").SGD
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_1x
dataloader = get_config("common/data/coco_panoptic_separated.py").dataloader
model = get_config("common/models/panoptic_fpn.py").model

model.backbone.bottom_up.freeze_at = 2
