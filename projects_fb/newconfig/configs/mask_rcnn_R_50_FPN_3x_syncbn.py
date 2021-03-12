from newconfig import ConfigFile

model = ConfigFile.load_rel("./models/mask_rcnn_fpn.py", "model")
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

dataloader = ConfigFile.load_rel("./data/coco.py")

lr_multiplier = ConfigFile.load_rel("./common_schedule.py", "lr_multiplier_3x")

optimizer = ConfigFile.load_rel("./common_optim.py", "SGD")

train = ConfigFile.load_rel("./common_train.py", "train")
train.max_iter = 270000  # 3x for batchsize = 16
