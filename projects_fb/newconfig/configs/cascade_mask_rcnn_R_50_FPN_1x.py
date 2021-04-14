from detectron2.model_zoo import get_config
from .mask_rcnn_R_50_FPN_1x import optimizer, lr_multiplier, train, dataloader

model = get_config("common/models/cascade_rcnn.py").model

model.backbone.bottom_up.freeze_at = 2
