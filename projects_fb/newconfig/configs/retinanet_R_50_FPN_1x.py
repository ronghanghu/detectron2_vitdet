# -*- coding: utf-8 -*-

from newconfig import ConfigFile

model = ConfigFile.load_rel("./models/base_retinanet.py", "model")
model.backbone.bottom_up.freeze_at = 2

dataloader = ConfigFile.load_rel("./data/coco.py")

lr_multiplier = ConfigFile.load_rel("./common_schedule.py", "lr_multiplier_1x")

optimizer = ConfigFile.load_rel("./common_optim.py", "SGD")
optimizer.lr = 0.01

train = ConfigFile.load_rel("./common_train.py", "train")
