from .mask_rcnn_R_50_FPN_1x import dataloader, model, optimizer, train
from .common_schedule import lr_multiplier_3x as lr_multiplier

model.backbone.bottom_up.freeze_at = 0
# fmt: off
model.backbone.bottom_up.stem.norm = \
    model.backbone.bottom_up.stages.norm = \
    model.backbone.norm = model.roi_heads.box_head.conv_norm = \
    model.roi_heads.mask_head.conv_norm = "SyncBN"
# fmt: on
# 4conv1fc head
model.roi_heads.box_head.conv_dims = [256, 256, 256, 256]
model.roi_heads.box_head.fc_dims = [1024]

train.max_iter = 270000  # 3x for batchsize = 16
