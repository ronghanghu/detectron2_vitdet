"""
This file contains utility functions for building detection models from
a configuration file.
"""

from torch_detectron.helpers.config_utils import ConfigClass
import torch_detectron.model_builder.generalized_rcnn as generalized_rcnn

from torch_detectron.core.resnet_builder import resnet50_conv4_body, resnet50_conv5_head
from torch_detectron.core.anchor_generator import AnchorGenerator, FPNAnchorGenerator
from torch_detectron.core.box_selector import RPNBoxSelector, FPNRPNBoxSelector, ROI2FPNLevelsMapper

from torch_detectron.core.proposal_matcher import Matcher
from torch_detectron.core.balanced_positive_negative_sampler import (
        BalancedPositiveNegativeSampler)

from torch_detectron.core.rpn_losses import (RPNLossComputation,
        RPNTargetPreparator)
from torch_detectron.core.fast_rcnn_losses import (FastRCNNLossComputation,
        FastRCNNTargetPreparator)
from torch_detectron.core.faster_rcnn import RPNHeads, Pooler
from torch_detectron.core.box_coder import BoxCoder
from torch_detectron.core.post_processor import PostProcessor, FPNPostProcessor

from torch_detectron.core.mask_rcnn_losses import (MaskRCNNLossComputation,
        MaskTargetPreparator)
from torch_detectron.core.mask_rcnn import MaskPostProcessor

class ModelBuilder(ConfigClass):
    def __call__(self):
        rpn_only = self.RPN_ONLY
        backbone = self.BACKBONE()
        self.REGION_PROPOSAL.RPN_ONLY = rpn_only
        region_proposal = self.REGION_PROPOSAL()

        self.HEADS.USE_MASK = self.USE_MASK
        heads = self.HEADS()
        return generalized_rcnn.GeneralizedRCNN(backbone, region_proposal, heads, rpn_only)


class BackboneBuilder(ConfigClass):
    def __call__(self):
        weights = self.WEIGHTS
        model_builder = self.BUILDER
        return model_builder(weights)


# FIXME use this or not
class AnchorGeneratorBuilder(ConfigClass):
    def __call__(self):
        scales = self.SCALES
        aspect_ratios = self.ASPECT_RATIOS
        base_anchor_size = self.BASE_ANCHOR_SIZE
        anchor_stride = self.ANCHOR_STRIDE
        anchor_generator = AnchorGenerator(scales=scales, aspect_ratios=aspect_ratios,
                base_anchor_size=base_anchor_size, anchor_stride=anchor_stride)
        return anchor_generator


class RPNBuilder(ConfigClass):
    def __call__(self):
        use_fpn = self.USE_FPN

        scales = self.SCALES
        aspect_ratios = self.ASPECT_RATIOS
        base_anchor_size = self.BASE_ANCHOR_SIZE
        anchor_stride = self.ANCHOR_STRIDE
        straddle_thresh = self.STRADDLE_THRESH

        anchor_maker = AnchorGenerator if not use_fpn else FPNAnchorGenerator
        anchor_args = {}
        # FIXME unify the args of AnchorGenerator and FPNAnchorGenerator?
        anchor_args['scales'] = scales
        anchor_args['aspect_ratios'] = aspect_ratios
        anchor_args['base_anchor_size'] = base_anchor_size
        anchor_args['straddle_thresh'] = straddle_thresh
        if use_fpn:
            anchor_args['anchor_strides'] = anchor_stride
        else:
            anchor_args['anchor_stride'] = anchor_stride
        anchor_generator = anchor_maker(**anchor_args)

        num_input_features = self.NUM_INPUT_FEATURES
        rpn_heads = RPNHeads(num_input_features, anchor_generator.num_anchors_per_location()[0])
        weights = self.WEIGHTS
        if weights:
            rpn_heads.load_state_dict(weights)

        box_selector_maker = RPNBoxSelector
        box_selector_args = {}
        if use_fpn:
            # TODO expose those options
            roi_to_fpn_level_mapper = ROI2FPNLevelsMapper(2, 5)
            box_selector_maker = FPNRPNBoxSelector
            box_selector_args['roi_to_fpn_level_mapper'] = roi_to_fpn_level_mapper
            fpn_post_nms_top_n = self.FPN_POST_NMS_TOP_N
            box_selector_args['fpn_post_nms_top_n'] = fpn_post_nms_top_n

        pre_nms_top_n = self.PRE_NMS_TOP_N
        post_nms_top_n = self.POST_NMS_TOP_N
        nms_thresh = self.NMS_THRESH
        min_size = self.MIN_SIZE
        box_selector_train = box_selector_maker(pre_nms_top_n=pre_nms_top_n,
                post_nms_top_n=post_nms_top_n,
                nms_thresh=nms_thresh, min_size=min_size, **box_selector_args)

        if use_fpn:
            fpn_post_nms_top_n = self.FPN_POST_NMS_TOP_N_TEST
            box_selector_args['fpn_post_nms_top_n'] = fpn_post_nms_top_n

        pre_nms_top_n_test = self.PRE_NMS_TOP_N_TEST
        post_nms_top_n_test = self.POST_NMS_TOP_N_TEST
        box_selector_test = box_selector_maker(pre_nms_top_n=pre_nms_top_n_test,
                post_nms_top_n=post_nms_top_n_test,
                nms_thresh=nms_thresh, min_size=min_size, **box_selector_args)

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

        rpn_only = self.RPN_ONLY

        module = generalized_rcnn.RPNModule(anchor_generator, rpn_heads,
                box_selector_train, box_selector_test, rpn_loss_evaluator, rpn_only)
        return module


class PoolerBuilder(ConfigClass):
    def __call__(self):
        module = self.MODULE
        pooler = Pooler(module)
        return pooler

class DetectionAndMaskHeadsBuilder(ConfigClass):
    def __call__(self):
        use_fpn = self.USE_FPN
        use_mask = self.USE_MASK

        pooler = self.POOLER()

        num_classes = self.NUM_CLASSES
        pretrained_weights = self.WEIGHTS
        module_builder = self.BUILDER

        # FIXME this is an ugly hack
        if use_mask and not use_fpn:
            classifier_layers = module_builder(pretrained_path=pretrained_weights)

            head_builder = self.HEAD_BUILDER
            classifier = head_builder(num_classes, pretrained_weights)
        else:
            classifier_layers = module_builder(num_classes=num_classes, pretrained=pretrained_weights)

        bbox_reg_weights = self.BBOX_REG_WEIGHTS
        box_coder = BoxCoder(weights=bbox_reg_weights)

        score_thresh = self.SCORE_THRESH
        nms_thresh = self.NMS
        detections_per_img = self.DETECTIONS_PER_IMG

        postprocessor_maker = PostProcessor if not use_fpn else FPNPostProcessor
        postprocessor = postprocessor_maker(score_thresh, nms_thresh, detections_per_img, box_coder)

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

        # mask

        if use_mask and not use_fpn:
            mask_builder = self.MASK_BUILDER
            heads_mask = mask_builder(num_classes, pretrained_weights)

            discretization_size = self.MASK_RESOLUTION
            mask_target_preparator = MaskTargetPreparator(matcher, discretization_size)
            mask_loss_evaluator = MaskRCNNLossComputation(mask_target_preparator)

            masker = None
            mask_post_processor = MaskPostProcessor(masker)

            return generalized_rcnn.DetectionAndMaskHead(pooler, classifier_layers,
                    postprocessor, loss_evaluator, classifier,
                    heads_mask, mask_loss_evaluator, mask_post_processor)

        if use_mask and use_fpn:
            mask_builder = self.MASK_BUILDER
            heads_mask = mask_builder(num_classes, pretrained_weights)

            mask_pooler = self.MASK_POOLER()

            discretization_size = self.MASK_RESOLUTION
            mask_target_preparator = MaskTargetPreparator(matcher, discretization_size)
            mask_loss_evaluator = MaskRCNNLossComputation(mask_target_preparator)

            masker = None
            mask_post_processor = MaskPostProcessor(masker)

            return generalized_rcnn.DetectionAndMaskFPNHead(pooler, classifier_layers,
                    postprocessor, loss_evaluator,
                    heads_mask, mask_pooler, mask_loss_evaluator, mask_post_processor)

        return generalized_rcnn.DetectionHead(pooler, classifier_layers,
                postprocessor, loss_evaluator)
