import torch

from detectron2.solver.lr_scheduler import WarmupMultiStepLR

from newconfig import ConfigFile
from newconfig import LazyCall as L

model = ConfigFile.load_rel("./models/base_cascade_rcnn.py", "model")

dataloader = ConfigFile.load_rel("./data/coco.py")

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

scheduler = L(WarmupMultiStepLR)(
    # optimizer=?
    # Both mmdet and classyvision use its own scheduler that does not depend on optimizer.
    # We've been wanting this for a long time https://github.com/fairinternal/detectron2/issues/184
    milestones=[60000, 80000],
    gamma=0.1,
)
