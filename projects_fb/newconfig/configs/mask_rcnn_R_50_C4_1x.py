from .common_optim import SGD as optimizer
from .common_schedule import lr_multiplier_1x as lr_multiplier
from .common_train import train
from .data.coco import dataloader
from .models.mask_rcnn_c4 import model

model.backbone.freeze_at = 2
