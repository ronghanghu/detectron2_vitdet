from newconfig import ConfigFile

model = ConfigFile.load_rel("./models/panoptic_fpn.py", "model")
model.backbone.bottom_up.freeze_at = 2

dataloader = ConfigFile.load_rel("./data/coco_panoptic_separated.py")

lr_multiplier = ConfigFile.load_rel("./common_schedule.py", "lr_multiplier_1x")

optimizer = ConfigFile.load_rel("./common_optim.py", "SGD")

train = ConfigFile.load_rel("./common_train.py", "train")