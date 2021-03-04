from newconfig import ConfigFile

model = ConfigFile.load_rel("./models/cascade_rcnn.py", "model")

dataloader = ConfigFile.load_rel("./data/coco.py")

lr_multiplier = ConfigFile.load_rel("./common_schedule.py", "lr_multiplier_1x")

optimizer = ConfigFile.load_rel("./common_optim.py", "SGD")

train = ConfigFile.load_rel("./common_train.py", "train")
