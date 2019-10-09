# -*- coding: utf-8 -*-
# File:


from . import model_zoo as _UNUSED  # register the handler
from .detection_checkpoint import DetectionCheckpointer
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer

__all__ = ["Checkpointer", "PeriodicCheckpointer", "DetectionCheckpointer"]
