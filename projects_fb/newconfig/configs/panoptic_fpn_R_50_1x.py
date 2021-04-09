from .common_optim import SGD as optimizer
from .common_schedule import lr_multiplier_1x as lr_multiplier
from .common_train import train
from .data.coco_panoptic_separated import dataloader
from .models.panoptic_fpn import model

model.backbone.bottom_up.freeze_at = 2
