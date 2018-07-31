"""
This file contains utility functions for building detection models from
a configuration file.
"""
import torch
from torch import nn

import torch_detectron.model_builder.generalized_rcnn as generalized_rcnn
import torch_detectron.model_builder.resnet as resnet
from torch_detectron.core.anchor_generator import AnchorGenerator
from torch_detectron.core.anchor_generator import FPNAnchorGenerator
from torch_detectron.core.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from torch_detectron.core.box_coder import BoxCoder
from torch_detectron.core.box_selector import FPNRPNBoxSelector
from torch_detectron.core.box_selector import ROI2FPNLevelsMapper
from torch_detectron.core.box_selector import RPNBoxSelector
from torch_detectron.core.fast_rcnn_losses import FastRCNNLossComputation
from torch_detectron.core.fast_rcnn_losses import FastRCNNTargetPreparator
from torch_detectron.core.faster_rcnn import Pooler
from torch_detectron.core.faster_rcnn import RPNHeads
from torch_detectron.core.fpn import FPN
from torch_detectron.core.fpn import FPNHeadClassifier
from torch_detectron.core.fpn import LastLevelMaxPool
from torch_detectron.core.mask_rcnn import MaskPostProcessor
from torch_detectron.core.mask_rcnn import MaskRCNNHeads
from torch_detectron.core.mask_rcnn_losses import MaskRCNNLossComputation
from torch_detectron.core.mask_rcnn_losses import MaskTargetPreparator
from torch_detectron.core.matcher import Matcher
from torch_detectron.core.post_processor import FPNPostProcessor
from torch_detectron.core.post_processor import PostProcessor
from torch_detectron.core.rpn_losses import RPNLossComputation
from torch_detectron.core.rpn_losses import RPNTargetPreparator
from torch_detectron.core.utils import load_state_dict
from torch_detectron.helpers.config_utils import ConfigNode


class ModelBuilder(ConfigNode):
    def __call__(self):
        rpn_only = self.config.MODEL.RPN_ONLY
        backbone = self.config.MODEL.BACKBONE()
        region_proposal = self.config.MODEL.RPN()
        roi_heads = None if rpn_only else self.config.MODEL.ROI_HEADS()
        return generalized_rcnn.GeneralizedRCNN(
            backbone, region_proposal, roi_heads, rpn_only
        )


class BackboneBuilder(ConfigNode):
    def __call__(self):
        weights = self.config.MODEL.BACKBONE.WEIGHTS
        model_builder = self.config.MODEL.BACKBONE.BUILDER
        return model_builder(self.config, pretrained=weights)


# FIXME use this or not
class AnchorGeneratorBuilder(ConfigNode):
    def __call__(self):
        scales = self.SCALES
        aspect_ratios = self.ASPECT_RATIOS
        base_anchor_size = self.BASE_ANCHOR_SIZE
        anchor_stride = self.ANCHOR_STRIDE
        anchor_generator = AnchorGenerator(
            scales=scales,
            aspect_ratios=aspect_ratios,
            base_anchor_size=base_anchor_size,
            anchor_stride=anchor_stride,
        )
        return anchor_generator


class RPNBuilder(ConfigNode):
    def __call__(self):
        use_fpn = self.config.MODEL.RPN.USE_FPN

        scales = self.config.MODEL.RPN.SCALES
        aspect_ratios = self.config.MODEL.RPN.ASPECT_RATIOS
        base_anchor_size = self.config.MODEL.RPN.BASE_ANCHOR_SIZE
        anchor_stride = self.config.MODEL.RPN.ANCHOR_STRIDE
        straddle_thresh = self.config.MODEL.RPN.STRADDLE_THRESH

        anchor_maker = AnchorGenerator if not use_fpn else FPNAnchorGenerator
        anchor_args = {}
        # FIXME unify the args of AnchorGenerator and FPNAnchorGenerator?
        anchor_args["scales"] = scales
        anchor_args["aspect_ratios"] = aspect_ratios
        anchor_args["base_anchor_size"] = base_anchor_size
        anchor_args["straddle_thresh"] = straddle_thresh
        if use_fpn:
            anchor_args["anchor_strides"] = anchor_stride
        else:
            anchor_args["anchor_stride"] = anchor_stride
        anchor_generator = anchor_maker(**anchor_args)

        num_input_features = self.config.MODEL.BACKBONE.OUTPUT_DIM
        rpn_heads = RPNHeads(
            num_input_features, anchor_generator.num_anchors_per_location()[0]
        )
        weights = self.config.MODEL.RPN.WEIGHTS
        if weights:
            load_state_dict(rpn_heads, weights)

        rpn_box_coder = BoxCoder(weights=(1., 1., 1., 1.))

        box_selector_maker = RPNBoxSelector
        box_selector_args = {}
        if use_fpn:
            # TODO expose those options
            roi_to_fpn_level_mapper = ROI2FPNLevelsMapper(2, 5)
            box_selector_maker = FPNRPNBoxSelector
            box_selector_args["roi_to_fpn_level_mapper"] = roi_to_fpn_level_mapper
            fpn_post_nms_top_n = self.FPN_POST_NMS_TOP_N
            box_selector_args["fpn_post_nms_top_n"] = fpn_post_nms_top_n

        pre_nms_top_n = self.config.MODEL.RPN.PRE_NMS_TOP_N
        post_nms_top_n = self.config.MODEL.RPN.POST_NMS_TOP_N
        nms_thresh = self.config.MODEL.RPN.NMS_THRESH
        min_size = self.config.MODEL.RPN.MIN_SIZE
        box_selector_train = box_selector_maker(
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=nms_thresh,
            min_size=min_size,
            box_coder=rpn_box_coder,
            **box_selector_args
        )

        if use_fpn:
            fpn_post_nms_top_n = self.config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST
            box_selector_args["fpn_post_nms_top_n"] = fpn_post_nms_top_n

        pre_nms_top_n_test = self.config.MODEL.RPN.PRE_NMS_TOP_N_TEST
        post_nms_top_n_test = self.config.MODEL.RPN.POST_NMS_TOP_N_TEST
        box_selector_test = box_selector_maker(
            pre_nms_top_n=pre_nms_top_n_test,
            post_nms_top_n=post_nms_top_n_test,
            nms_thresh=nms_thresh,
            min_size=min_size,
            box_coder=rpn_box_coder,
            **box_selector_args
        )

        # loss evaluation
        fg_iou_threshold = self.config.MODEL.RPN.FG_IOU_THRESHOLD
        bg_iou_threshold = self.config.MODEL.RPN.BG_IOU_THRESHOLD
        rpn_matcher = Matcher(
            fg_iou_threshold, bg_iou_threshold, allow_low_quality_matches=True
        )

        rpn_target_preparator = RPNTargetPreparator(rpn_matcher, rpn_box_coder)

        batch_size_per_image = self.config.MODEL.RPN.BATCH_SIZE_PER_IMAGE
        positive_fraction = self.config.MODEL.RPN.POSITIVE_FRACTION
        rpn_fg_bg_sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image=batch_size_per_image,
            positive_fraction=positive_fraction,
        )
        rpn_loss_evaluator = RPNLossComputation(
            rpn_target_preparator, rpn_fg_bg_sampler
        )

        rpn_only = self.config.MODEL.RPN_ONLY

        module = generalized_rcnn.RPNModule(
            anchor_generator,
            rpn_heads,
            box_selector_train,
            box_selector_test,
            rpn_loss_evaluator,
            rpn_only,
        )
        return module


class PoolerBuilder(ConfigNode):
    def __call__(self):
        module = self.config.MODEL.ROI_HEADS.POOLER.MODULE
        pooler = Pooler(module)
        return pooler


class DetectionAndMaskHeadsBuilder(ConfigNode):
    def __call__(self):
        use_fpn = self.config.MODEL.ROI_HEADS.USE_FPN
        use_mask = self.config.MODEL.USE_MASK

        pooler = self.config.MODEL.ROI_HEADS.POOLER()

        num_classes = self.config.MODEL.ROI_HEADS.NUM_CLASSES
        pretrained_weights = self.config.MODEL.ROI_HEADS.WEIGHTS
        module_builder = self.config.MODEL.ROI_HEADS.BUILDER

        # FIXME this is an ugly hack
        if use_mask and not use_fpn:
            classifier_layers = module_builder(
                self.config, pretrained_path=pretrained_weights
            )

            head_builder = self.config.MODEL.ROI_HEADS.HEAD_BUILDER
            classifier = head_builder(self.config, num_classes, pretrained_weights)
        else:
            classifier_layers = module_builder(
                self.config, num_classes=num_classes, pretrained=pretrained_weights
            )

        bbox_reg_weights = self.config.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
        box_coder = BoxCoder(weights=bbox_reg_weights)

        score_thresh = self.config.MODEL.ROI_HEADS.SCORE_THRESH
        nms_thresh = self.config.MODEL.ROI_HEADS.NMS
        detections_per_img = self.config.MODEL.ROI_HEADS.DETECTIONS_PER_IMG

        postprocessor_maker = PostProcessor if not use_fpn else FPNPostProcessor
        postprocessor = postprocessor_maker(
            score_thresh, nms_thresh, detections_per_img, box_coder
        )

        # loss evaluation
        fg_iou_threshold = self.config.MODEL.ROI_HEADS.FG_IOU_THRESHOLD
        bg_iou_threshold = self.config.MODEL.ROI_HEADS.BG_IOU_THRESHOLD
        matcher = Matcher(fg_iou_threshold, bg_iou_threshold)

        target_preparator = FastRCNNTargetPreparator(matcher, box_coder)

        batch_size_per_image = self.config.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        positive_fraction = self.config.MODEL.ROI_HEADS.POSITIVE_FRACTION
        fg_bg_sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image=batch_size_per_image,
            positive_fraction=positive_fraction,
        )
        loss_evaluator = FastRCNNLossComputation(target_preparator, fg_bg_sampler)

        # mask

        if use_mask and not use_fpn:
            mask_builder = self.config.MODEL.ROI_HEADS.MASK_BUILDER
            heads_mask = mask_builder(self.config, num_classes, pretrained_weights)

            discretization_size = self.config.MODEL.ROI_HEADS.MASK_RESOLUTION
            mask_target_preparator = MaskTargetPreparator(matcher, discretization_size)
            mask_loss_evaluator = MaskRCNNLossComputation(mask_target_preparator)

            masker = None
            mask_post_processor = MaskPostProcessor(masker)

            return generalized_rcnn.DetectionAndMaskHead(
                pooler,
                classifier_layers,
                postprocessor,
                loss_evaluator,
                classifier,
                heads_mask,
                mask_loss_evaluator,
                mask_post_processor,
            )

        if use_mask and use_fpn:
            # TODO there are a number of things that are implicit here.
            # for example, the mask_pooler should be subsampling the positive
            # boxes only, so that the loss evaluator knows that it should do the same
            mask_builder = self.config.MODEL.ROI_HEADS.MASK_BUILDER
            heads_mask = mask_builder(self.config, num_classes, pretrained_weights)

            mask_pooler = self.config.MODEL.ROI_HEADS.MASK_POOLER()

            discretization_size = self.config.MODEL.ROI_HEADS.MASK_RESOLUTION
            mask_target_preparator = MaskTargetPreparator(matcher, discretization_size)
            mask_loss_evaluator = MaskRCNNLossComputation(
                mask_target_preparator, subsample_only_positive_boxes=True
            )

            masker = None
            mask_post_processor = MaskPostProcessor(masker)

            return generalized_rcnn.DetectionAndMaskFPNHead(
                pooler,
                classifier_layers,
                postprocessor,
                loss_evaluator,
                heads_mask,
                mask_pooler,
                mask_loss_evaluator,
                mask_post_processor,
            )

        return generalized_rcnn.DetectionHead(
            pooler, classifier_layers, postprocessor, loss_evaluator
        )


def resnet50_conv4_body(config, pretrained=None):
    model = resnet.ResNetBackbone(
        stem_function=config.MODEL.RESNET.STEM_FUNCTION,
        block_module=config.MODEL.RESNET.BLOCK_MODULE,
        stages=resnet.ResNet50StagesTo4,
        num_groups=config.MODEL.RESNET.NUM_GROUPS,
        width_per_group=config.MODEL.RESNET.WIDTH_PER_GROUP,
        stride_in_1x1=config.MODEL.RESNET.STRIDE_IN_1X1,
    )
    if pretrained:
        state_dict = torch.load(pretrained)
        load_state_dict(model, state_dict, strict=False)
    return model


def resnet50_conv5_head(config, num_classes, pretrained=None):
    stage = resnet.StageSpec(index=5, block_count=3, return_features=False)
    head = resnet.ResNetHead(
        block_module=config.MODEL.RESNET.BLOCK_MODULE,
        stages=(stage,),
        num_groups=config.MODEL.RESNET.NUM_GROUPS,
        width_per_group=config.MODEL.RESNET.WIDTH_PER_GROUP,
        stride_in_1x1=config.MODEL.RESNET.STRIDE_IN_1X1,
    )
    classifier = resnet.ClassifierHead(2048, num_classes)
    if pretrained:
        state_dict = torch.load(pretrained)
        load_state_dict(head, state_dict, strict=False)
        load_state_dict(classifier, state_dict, strict=False)
    model = nn.Sequential(head, classifier)
    return model


def resnet50_conv5_body(config):
    model = resnet.ResNetBackbone(
        stem_function=config.MODEL.RESNET.STEM_FUNCTION,
        block_module=config.MODEL.RESNET.BLOCK_MODULE,
        stages=resnet.ResNet50StagesTo5,
        num_groups=config.MODEL.RESNET.NUM_GROUPS,
        width_per_group=config.MODEL.RESNET.WIDTH_PER_GROUP,
        stride_in_1x1=config.MODEL.RESNET.STRIDE_IN_1X1,
    )
    return model


def fpn_resnet50_conv5_body(config, pretrained=None):
    body = resnet.ResNetBackbone(
        stem_function=config.MODEL.RESNET.STEM_FUNCTION,
        block_module=config.MODEL.RESNET.BLOCK_MODULE,
        stages=resnet.ResNet50FPNStagesTo5,
        num_groups=config.MODEL.RESNET.NUM_GROUPS,
        width_per_group=config.MODEL.RESNET.WIDTH_PER_GROUP,
        stride_in_1x1=config.MODEL.RESNET.STRIDE_IN_1X1,
    )
    representation_size = config.MODEL.BACKBONE.OUTPUT_DIM
    fpn = FPN(
        layers=[256, 512, 1024, 2048],
        representation_size=representation_size,
        top_blocks=LastLevelMaxPool(),
    )
    if pretrained:
        state_dict = torch.load(pretrained)
        load_state_dict(body, state_dict, strict=False)
        load_state_dict(fpn, state_dict, strict=False)
    model = nn.Sequential(body, fpn)
    return model


def fpn_classification_head(config, num_classes, pretrained=None):
    representation_size = config.MODEL.BACKBONE.OUTPUT_DIM
    model = FPNHeadClassifier(num_classes, representation_size * 7 * 7, 1024)
    if pretrained:
        state_dict = torch.load(pretrained)
        load_state_dict(model, state_dict, strict=False)
    return model


def maskrcnn_head(config, num_classes, pretrained=None):
    model = MaskRCNNHeads(256, [256, 256, 256, 256], num_classes)
    if pretrained:
        state_dict = torch.load(pretrained)
        load_state_dict(model, state_dict, strict=False)
    return model
