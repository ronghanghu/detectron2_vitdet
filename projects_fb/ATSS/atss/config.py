# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


def add_atss_config(cfg):
    """
    Add config for ATSS.
    """
    cfg.MODEL.ATSS = cfg.MODEL.RETINANET
    cfg.MODEL.ATSS.TOPK = 9
    # One of three options ("", "iou", "distance")
    # Where iou and distance simply take the topk candidates
    # Instead of using mean + stddev as threshold
    cfg.MODEL.ATSS.SELECTION_MODE = ""
    cfg.MODEL.ATSS.REG_LOSS_WEIGHT = 2.0
