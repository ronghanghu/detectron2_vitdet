# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from detectron2.config import CfgNode as CN


def add_masktrack_r_cnn_config(cfg):
    """
    Add config for MaskTrack-R-CNN
    """
    cfg.SOLVER.OPTIMIZER = "SGD"
    cfg.MODEL.TRACK_ON = True
    cfg.MODEL.ROI_TRACK_HEAD = CN()
    cfg.MODEL.ROI_TRACK_HEAD.NAME = "TrackHead"
    cfg.MODEL.ROI_TRACK_HEAD.NUM_FCS = 2
    cfg.MODEL.ROI_TRACK_HEAD.IN_CHANNELS = 256
    cfg.MODEL.ROI_TRACK_HEAD.FC_OUT_CHANNELS = 1024
    cfg.MODEL.ROI_TRACK_HEAD.ROI_FEAT_SIZE = 7
    cfg.MODEL.ROI_TRACK_HEAD.MATCH_COEFF = [1.0, 2.0, 10]
