from .common_optim import SGD as optimizer
from .common_schedule import lr_multiplier_1x as lr_multiplier
from .common_train import train
from .data.coco import dataloader
from .models.mask_rcnn_fpn import model

# equivalent to:
# model = ConfigFile.load_rel("./models/mask_rcnn_fpn.py", "model")

model.backbone.bottom_up.freeze_at = 2
