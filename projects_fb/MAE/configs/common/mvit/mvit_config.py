# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from detectron2.config import CfgNode as CN


def mvit_config():
    """
    Add config for tridentnet.
    """
    _C = CN()

    # -----------------------------------------------------------------------------
    # MViT options
    # -----------------------------------------------------------------------------
    _C.MVIT = CN()

    # Options include `conv`, `max`.
    _C.MVIT.MODE = "conv"

    # If True, perform pool before projection in attention.
    _C.MVIT.POOL_FIRST = False

    # If True, use cls embed in the network, otherwise don't use cls_embed in transformer.
    _C.MVIT.CLS_EMBED_ON = True

    # Kernel size for patchtification.
    _C.MVIT.PATCH_KERNEL = [3, 7, 7]

    # Stride size for patchtification.
    _C.MVIT.PATCH_STRIDE = [2, 4, 4]

    # Padding size for patchtification.
    _C.MVIT.PATCH_PADDING = [2, 4, 4]

    # If True, use 2d patch, otherwise use 3d patch.
    _C.MVIT.PATCH_2D = False

    # Base embedding dimension for the transformer.
    _C.MVIT.EMBED_DIM = 96

    # Base num of heads for the transformer.
    _C.MVIT.NUM_HEADS = 1

    # Dimension reduction ratio for the MLP layers.
    _C.MVIT.MLP_RATIO = 4.0

    # If use, use bias term in attention fc layers.
    _C.MVIT.QKV_BIAS = True

    # Drop path rate for the tranfomer.
    _C.MVIT.DROPPATH_RATE = 0.1

    # Depth of the transformer.
    _C.MVIT.DEPTH = 16

    # Normalization layer for the transformer. Only layernorm is supported now.
    _C.MVIT.NORM = "layernorm"

    # Dimension multiplication at layer i. If 2.0 is used, then the next block will increase
    # the dimension by 2 times. Format: [depth_i: mul_dim_ratio]
    _C.MVIT.DIM_MUL = []

    # Head number multiplication at layer i. If 2.0 is used, then the next block will
    # increase the number of heads by 2 times. Format: [depth_i: head_mul_ratio]
    _C.MVIT.HEAD_MUL = []

    # Stride size for the Pool KV at layer i.
    # Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
    _C.MVIT.POOL_KV_STRIDE = []

    # Initial stride size for KV at layer 1. The stride size will be further reduced with
    # the raio of MVIT.DIM_MUL. If will overwrite MVIT.POOL_KV_STRIDE if not None.
    _C.MVIT.POOL_KV_STRIDE_ADAPTIVE = None

    # Stride size for the Pool Q at layer i.
    # Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
    _C.MVIT.POOL_Q_STRIDE = []

    # If not None, overwrite the KV_KERNEL and Q_KERNEL size with POOL_KVQ_CONV_SIZ.
    # Otherwise the kernel_size is [s + 1 if s > 1 else s for s in stride_size].
    _C.MVIT.POOL_KVQ_KERNEL = None

    # If True, perform no decay on positional embedding and cls embedding.
    _C.MVIT.ZERO_DECAY_POS_CLS = True

    # If True, use norm after stem.
    _C.MVIT.NORM_STEM = False

    # If True, perform separate positional embedding.
    _C.MVIT.SEP_POS_EMBED = False

    # Dropout rate for the MViT backbone.
    _C.MVIT.DROPOUT_RATE = 0.0

    # If True, use absolute positional embedding.
    _C.MVIT.USE_ABS_POS = True

    # If True, use relative positional embedding for spatial dimentions
    _C.MVIT.REL_POS_SPATIAL = False

    # If True, use relative positional embedding for temporal dimentions
    _C.MVIT.REL_POS_TEMPORAL = False

    # window size
    _C.MVIT.WIN_SIZE = [1, 56, 56]

    # window size strides
    _C.MVIT.WIN_SIZE_STRIDE = []

    # Activation checkpointing enabled or not to save GPU memory.
    _C.MVIT.ACT_CHECKPOINT = False

    # Freeze backbone. #1 for stem
    _C.MVIT.FREEZE_AT = 0

    # Freeze abs pos embed
    _C.MVIT.FREEZE_POS = False

    # new d2 configs
    # pre-train image size
    _C.MVIT.IMAGE_SIZE = 224
    _C.MVIT.OUT_FEATURES = ["scale2", "scale3", "scale4", "scale5"]

    # _C.SOLVER.OPTIMIZER = "SGD"
    # _C.SOLVER.ZERO_WEIGHT_DECAY = []

    # # random_select crop_augs. See detr augs.
    # _C.INPUT.CROP.RANDOM_SELECT = False

    return _C


config = mvit_config()
