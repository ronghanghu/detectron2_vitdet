import torch

from newconfig import ConfigFile
from newconfig import LazyCall as L

model = ConfigFile.load_rel("./models/base_cascade_rcnn.py", "model")

dataloader = ConfigFile.load_rel("./data/coco.py")

lr_multiplier = ConfigFile.load_rel("./common_schedule.py", "lr_multiplier_1x")

optimizer = L(torch.optim.SGD)(
    lr=0.02,
    momentum=0.9,
    # params=D(
    # type=get_default_optimizer_params,
    # model=
    # )
    # Both mmdet and classyvision use "optimizer factory" that does not
    # depend on models
)
