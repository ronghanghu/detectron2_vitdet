# -*- coding: utf-8 -*-

from .model_builder import META_ARCH_REGISTRY, build_model  # isort:skip

# import all the meta_arch, so they will be registered
from .rcnn import GeneralizedRCNN, ProposalNetwork
from .retinanet import RetinaNet
from .segmentor import PanopticFPN, SemanticSegmentor
