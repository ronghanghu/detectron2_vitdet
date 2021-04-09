# -*- coding: utf-8 -*-

from .models.retinanet import model

model.backbone.bottom_up.freeze_at = 2

from .data.coco import dataloader

from .common_schedule import lr_multiplier_1x as lr_multiplier

from .common_optim import SGD as optimizer

optimizer.lr = 0.01

from .common_train import train
