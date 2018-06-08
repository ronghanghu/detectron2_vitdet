"""
This file contains utility functions for building detection models from
a configuration file.
"""

from torch_detectron.helpers.config_utils import ConfigClass
import torch_detectron.model_builder.generalized_rcnn as generalized_rcnn

from torch_detectron.core.resnet_builder import resnet50_conv4_body, resnet50_conv5_head
from torch_detectron.core.anchor_generator import AnchorGenerator
from torch_detectron.core.box_selector import RPNBoxSelector

from torch_detectron.core.proposal_matcher import Matcher
from torch_detectron.core.balanced_positive_negative_sampler import (
        BalancedPositiveNegativeSampler)

from torch_detectron.core.rpn_losses import (RPNLossComputation,
        RPNTargetPreparator)
from torch_detectron.core.fast_rcnn_losses import (FastRCNNLossComputation,
        FastRCNNTargetPreparator)
from torch_detectron.core.faster_rcnn import RPNHeads, Pooler
from torch_detectron.core.box_coder import BoxCoder
from core.post_processor import PostProcessor


class ModelBuilder(ConfigClass):
    def __call__(self):
        backbone = self.BACKBONE()
        region_proposal = self.REGION_PROPOSAL()
        heads = self.HEADS()
        return generalized_rcnn.GeneralizedRCNN(backbone, region_proposal, heads)


class BackboneBuilder(ConfigClass):
    def __call__(self):
        weights = self.WEIGHTS
        model_builder = self.BUILDER
        return model_builder(weights)


class RPNBuilder(ConfigClass):
    def __call__(self):
        scales = self.SCALES
        aspect_ratios = self.ASPECT_RATIOS
        base_anchor_size = self.BASE_ANCHOR_SIZE
        anchor_stride = self.ANCHOR_STRIDE
        anchor_generator = AnchorGenerator(scales=scales, aspect_ratios=aspect_ratios,
                base_anchor_size=base_anchor_size, anchor_stride=anchor_stride)

        num_input_features = self.NUM_INPUT_FEATURES
        rpn_heads = RPNHeads(num_input_features, anchor_generator.num_anchors_per_location()[0])

        pre_nms_top_n = self.PRE_NMS_TOP_N
        post_nms_top_n = self.POST_NMS_TOP_N
        nms_thresh = self.NMS_THRESH
        min_size = self.MIN_SIZE
        box_selector_train = RPNBoxSelector(pre_nms_top_n, post_nms_top_n,
                nms_thresh, min_size)

        pre_nms_top_n_test = self.PRE_NMS_TOP_N_TEST
        post_nms_top_n_test = self.POST_NMS_TOP_N_TEST
        box_selector_test = RPNBoxSelector(pre_nms_top_n_test, post_nms_top_n_test,
                nms_thresh, min_size)

        # loss evaluation
        matched_threshold = self.MATCHED_THRESHOLD
        unmatched_threshold = self.UNMATCHED_THRESHOLD
        rpn_matcher = Matcher(matched_threshold, unmatched_threshold, force_match_for_each_row=True)
        rpn_box_coder = BoxCoder(weights=(1., 1., 1., 1.))
        rpn_target_preparator = RPNTargetPreparator(rpn_matcher, rpn_box_coder)

        batch_size_per_image = self.BATCH_SIZE_PER_IMAGE
        positive_fraction = self.POSITIVE_FRACTION
        rpn_fg_bg_sampler = BalancedPositiveNegativeSampler(
                batch_size_per_image=batch_size_per_image, positive_fraction=positive_fraction)
        rpn_loss_evaluator = RPNLossComputation(rpn_target_preparator, rpn_fg_bg_sampler)

        rpn_only = False

        module = generalized_rcnn.RPNModule(anchor_generator, rpn_heads,
                box_selector_train, box_selector_test, rpn_loss_evaluator, rpn_only)
        return module


class PoolerBuilder(ConfigClass):
    def __call__(self):
        module = self.MODULE
        pooler = Pooler(module)
        return pooler

class DetectionHeadBuilder(ConfigClass):
    def __call__(self):
        pooler = self.POOLER()

        num_classes = self.NUM_CLASSES
        pretrained_weights = self.WEIGHTS
        module_builder = self.BUILDER
        classifier_layers = module_builder(num_classes=num_classes, pretrained=pretrained_weights)

        bbox_reg_weights = self.BBOX_REG_WEIGHTS
        box_coder = BoxCoder(weights=bbox_reg_weights)

        score_thresh = self.SCORE_THRESH
        nms_thresh = self.NMS
        detections_per_img = self.DETECTIONS_PER_IMG
        postprocessor = PostProcessor(score_thresh, nms_thresh, detections_per_img, box_coder)

        # loss evaluation
        matched_threshold = self.MATCHED_THRESHOLD
        unmatched_threshold = self.UNMATCHED_THRESHOLD
        matcher = Matcher(matched_threshold, unmatched_threshold)

        target_preparator = FastRCNNTargetPreparator(matcher, box_coder)

        batch_size_per_image = self.BATCH_SIZE_PER_IMAGE
        positive_fraction = self.POSITIVE_FRACTION
        fg_bg_sampler = BalancedPositiveNegativeSampler(
                batch_size_per_image=batch_size_per_image,
                positive_fraction=positive_fraction)
        loss_evaluator = FastRCNNLossComputation(target_preparator, fg_bg_sampler)

        return generalized_rcnn.DetectionHead(pooler, classifier_layers, postprocessor, loss_evaluator)
