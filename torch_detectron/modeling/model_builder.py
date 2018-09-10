"""
Proposal implementation for the model builder
Everything that constructs something is a function that is
or can be accessed by a string
"""

import torch
from torch import nn


from ..structures.image_list import to_image_list


from torch_detectron.modeling.anchor_generator import (
    AnchorGenerator,
    FPNAnchorGenerator,
)
from torch_detectron.modeling.box_selector import RPNBoxSelector, FPNRPNBoxSelector

from torch_detectron.modeling.box_selector import ROI2FPNLevelsMapper

from torch_detectron.modeling.box_coder import BoxCoder
from torch_detectron.modeling.faster_rcnn import RPNPredictor

from torch_detectron.modeling.matcher import Matcher
from torch_detectron.modeling.rpn_losses import RPNTargetPreparator, RPNLossComputation

from torch_detectron.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)

from torch_detectron.modeling.fast_rcnn_losses import (
    FastRCNNTargetPreparator,
    FastRCNNLossComputation,
)

from torch_detectron.modeling.mask_rcnn_losses import (
    MaskTargetPreparator,
    MaskRCNNLossComputation,
)

from torch_detectron.modeling.post_processor import PostProcessor, FPNPostProcessor
from torch_detectron.modeling.mask_rcnn import MaskPostProcessor

from torch_detectron.modeling.utils import cat_bbox

from torch_detectron.modeling.roi_box_feature_extractors import (
    make_roi_box_feature_extractor
)
from torch_detectron.modeling.roi_box_predictors import make_roi_box_predictor
from torch_detectron.modeling.roi_mask_feature_extractors import (
    make_roi_mask_feature_extractor
)
from torch_detectron.modeling.roi_mask_predictors import make_roi_mask_predictor
from torch_detectron.modeling.backbones import build_backbone


class GeneralizedRCNN(nn.Module):
    """
    Main class for generalized r-cnn. Supports boxes, masks and keypoints
    This is very similar to what we had before, the difference being that now
    we construct the modules in __init__, instead of passing them as arguments
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.cfg = cfg.clone()

        # not implemented yet, but follows exactly Ross' implementation
        self.backbone = build_backbone(cfg)

        self.rpn = build_rpn(cfg)

        # individually create the heads, that will be combined together
        # afterwards
        roi_heads = []
        if not cfg.MODEL.RPN_ONLY:
            roi_heads.append(build_roi_box_head(cfg))
        if cfg.MODEL.MASK_ON:
            roi_heads.append(build_roi_mask_head(cfg))

        # combine individual heads in a single module
        if roi_heads:
            self.roi_heads = combine_roi_heads(cfg, roi_heads)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BBox]): ground-truth boxes present in the image (optional)
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.cfg.MODEL.RPN_ONLY:
            x = features
            result = proposals
            detector_losses = {}
        else:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result


_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)


################################################################################
# Those generic classes shows that it's possible to plug both Mask R-CNN C4
# and Mask R-CNN FPN with generic containers
################################################################################
class CascadedHeads(torch.nn.Module):
    """
    For Mask R-CNN FPN
    """

    def __init__(self, heads):
        super(CascadedHeads, self).__init__()
        self.heads = torch.nn.ModuleList(heads)

    def forward(self, features, proposals, targets=None):
        losses = {}
        for head in self.heads:
            x, proposals, loss = head(features, proposals, targets)
            losses.update(loss)

        return x, proposals, losses


class SharedROIHeads(torch.nn.Module):
    """
    For Mask R-CNN C4
    """

    def __init__(self, heads):
        super(SharedROIHeads, self).__init__()
        self.heads = torch.nn.ModuleList(heads)
        # manually share feature extractor
        shared_feature_extractor = self.heads[0].feature_extractor
        for head in self.heads[1:]:
            head.feature_extractor = shared_feature_extractor

    def forward(self, features, proposals, targets=None):
        losses = {}
        for head in self.heads:
            x, proposals, loss = head(features, proposals, targets)
            # during training, share the features
            if self.training:
                features = x
            losses.update(loss)

        return x, proposals, losses


def combine_roi_heads(cfg, roi_heads):
    """
    This function takes a list of modules for the rois computation and
    combines them into a single head module.
    It also support feature sharing, for example during
    Mask R-CNN C4.
    """
    constructor = CascadedHeads
    if cfg.MODEL.SHARE_FEATURES_DURING_TRAINING:
        constructor = SharedROIHeads
    # can also use getfunc to query a function by name
    return constructor(roi_heads)


################################################################################
# Create RPN
################################################################################
def make_anchor_generator(config):
    use_fpn = config.MODEL.RPN.USE_FPN

    scales = config.MODEL.RPN.SCALES
    aspect_ratios = config.MODEL.RPN.ASPECT_RATIOS
    base_anchor_size = config.MODEL.RPN.BASE_ANCHOR_SIZE
    anchor_stride = config.MODEL.RPN.ANCHOR_STRIDE
    straddle_thresh = config.MODEL.RPN.STRADDLE_THRESH

    anchor_maker = AnchorGenerator if not use_fpn else FPNAnchorGenerator
    anchor_args = {}
    # FIXME unify the args of AnchorGenerator and FPNAnchorGenerator?
    anchor_args["scales"] = scales
    anchor_args["aspect_ratios"] = aspect_ratios
    anchor_args["base_anchor_size"] = base_anchor_size
    anchor_args["straddle_thresh"] = straddle_thresh
    if use_fpn:
        anchor_args["anchor_strides"] = anchor_stride
        assert len(anchor_stride) == len(scales), "FPN should have len(ANCHOR_STRIDE) == len(SCALES)"
    else:
        assert len(anchor_sride) == 1, "Non-FPN should have a single ANCHOR_STRIDE"
        anchor_args["anchor_stride"] = anchor_stride[0]
    anchor_generator = anchor_maker(**anchor_args)
    return anchor_generator


def make_box_selector(config, rpn_box_coder, is_train):
    use_fpn = config.MODEL.RPN.USE_FPN
    box_selector_maker = RPNBoxSelector
    box_selector_args = {}
    if use_fpn:
        # TODO expose those options
        roi_to_fpn_level_mapper = ROI2FPNLevelsMapper(2, 5)
        box_selector_maker = FPNRPNBoxSelector
        box_selector_args["roi_to_fpn_level_mapper"] = roi_to_fpn_level_mapper
        fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N
        if not is_train:
            fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST
        box_selector_args["fpn_post_nms_top_n"] = fpn_post_nms_top_n

    pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N
    post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N
    if not is_train:
        pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TEST
        post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TEST
    nms_thresh = config.MODEL.RPN.NMS_THRESH
    min_size = config.MODEL.RPN.MIN_SIZE
    box_selector = box_selector_maker(
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        box_coder=rpn_box_coder,
        **box_selector_args
    )
    return box_selector


class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg):
        super(RPNModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator(cfg)

        num_input_features = cfg.BACKBONE.OUTPUT_DIM
        heads = RPNPredictor(
            num_input_features, anchor_generator.num_anchors_per_location()[0]
        )

        rpn_box_coder = BoxCoder(weights=(1., 1., 1., 1.))

        box_selector_train = make_box_selector(cfg, rpn_box_coder, is_train=True)
        box_selector_test = make_box_selector(cfg, rpn_box_coder, is_train=False)

        loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder)

        self.anchor_generator = anchor_generator
        self.heads = heads
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BBox): ground-truth boxes present in the image (optional)
        """
        objectness, rpn_box_regression = self.heads(features)
        anchors = self.anchor_generator(images.image_sizes, features)

        if not self.training:
            boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
            if self.cfg.MODEL.RPN_ONLY:
                # concatenate all boxes from different levels if in inference and rpn_only
                boxes = list(zip(*boxes))
                boxes = [cat_bbox(box) for box in boxes]
                # sort scores in decreasing order
                inds = [
                    box.get_field("objectness").sort(descending=True)[1]
                    for box in boxes
                ]
                boxes = [box[ind] for box, ind in zip(boxes, inds)]
            return boxes, {}

        boxes = anchors
        if not self.cfg.MODEL.RPN_ONLY:
            with torch.no_grad():
                boxes = self.box_selector_train(anchors, objectness, rpn_box_regression)
        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
            anchors, objectness, rpn_box_regression, targets
        )
        return (
            boxes,
            dict(loss_objectness=loss_objectness, loss_rpn_box_reg=loss_rpn_box_reg),
        )


def build_rpn(cfg):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    return RPNModule(cfg)


################################################################################
# RoiBoxHead
################################################################################


def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG

    postprocessor_maker = PostProcessor if not use_fpn else FPNPostProcessor
    postprocessor = postprocessor_maker(
        score_thresh, nms_thresh, detections_per_img, box_coder
    )
    return postprocessor


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        self.predictor = make_roi_box_predictor(cfg)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        if self.training:
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        x = self.feature_extractor(features, proposals)
        class_logits, box_regression = self.predictor(x)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )


def build_roi_box_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg)


################################################################################
# RoiMaskHead
################################################################################


def make_roi_mask_post_processor(cfg):
    masker = None
    mask_post_processor = MaskPostProcessor(masker)
    return mask_post_processor


class ROIMaskHead(torch.nn.Module):
    def __init__(self, cfg):
        super(ROIMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_mask_feature_extractor(cfg)
        self.predictor = make_roi_mask_predictor(cfg)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        # the network can't handle the case of no selected proposals,
        # so need to shortcut before
        # TODO handle this properly
        if not self.training:
            if sum(r.bbox.shape[0] for r in proposals) == 0:
                for r in proposals:
                    r.add_field("mask", features[0].new())
                return features, proposals, {}

        if self.training and self.cfg.MODEL.SHARE_FEATURES_DURING_TRAINING:
            x = features
        else:
            if self.cfg.MODEL.SHARE_FEATURES_DURING_TRAINING:
                # proposals have been flattened by this point, so aren't
                # in the format of list[list[BBox]] anymore. Add one more level to it
                proposals = [proposals]
            x = self.feature_extractor(features, proposals)
        mask_logits = self.predictor(x)

        if not self.training:
            # TODO Fix this horrible hack
            if self.cfg.MODEL.SHARE_FEATURES_DURING_TRAINING:
                proposals = proposals[0]
            result = self.post_processor(mask_logits, proposals)
            return x, result, {}

        loss_mask = self.loss_evaluator(proposals, mask_logits, targets)

        return x, proposals, dict(loss_mask=loss_mask)


def build_roi_mask_head(cfg):
    return ROIMaskHead(cfg)


################################################################################
# functions for creating loss evaluators. The structore of loss evaluators
# might change, but is here for illustrative purposes
################################################################################
def make_standard_loss_evaluator(
    loss_type,
    fg_iou_threshold,
    bg_iou_threshold,
    batch_size_per_image=None,
    positive_fraction=None,
    box_coder=None,
    mask_resolution=None,
    mask_subsample_only_positive_boxes=None,
):
    assert loss_type in ("rpn", "fast_rcnn", "mask_rcnn")
    allow_low_quality_matches = loss_type == "rpn"
    matcher = Matcher(
        fg_iou_threshold,
        bg_iou_threshold,
        allow_low_quality_matches=allow_low_quality_matches,
    )

    if loss_type in ("rpn", "fast_rcnn"):
        assert isinstance(batch_size_per_image, int)
        assert isinstance(positive_fraction, (int, float))
        assert isinstance(box_coder, BoxCoder)
        fg_bg_sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image=batch_size_per_image,
            positive_fraction=positive_fraction,
        )

    if loss_type == "rpn":
        for arg in (mask_resolution, mask_subsample_only_positive_boxes):
            assert arg is None
        target_preparator = RPNTargetPreparator(matcher, box_coder)
        loss_evaluator = RPNLossComputation(target_preparator, fg_bg_sampler)
    elif loss_type == "fast_rcnn":
        for arg in (mask_resolution, mask_subsample_only_positive_boxes):
            assert arg is None
        target_preparator = FastRCNNTargetPreparator(matcher, box_coder)
        loss_evaluator = FastRCNNLossComputation(target_preparator, fg_bg_sampler)
    elif loss_type == "mask_rcnn":
        for arg in (batch_size_per_image, positive_fraction):
            assert arg is None
        assert isinstance(mask_resolution, (int, float))
        assert isinstance(mask_subsample_only_positive_boxes, bool)
        target_preparator = MaskTargetPreparator(matcher, mask_resolution)
        loss_evaluator = MaskRCNNLossComputation(
            target_preparator,
            subsample_only_positive_boxes=mask_subsample_only_positive_boxes,
        )

    return loss_evaluator


def make_rpn_loss_evaluator(cfg, rpn_box_coder):
    return make_standard_loss_evaluator(
        "rpn",
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
        cfg.MODEL.RPN.POSITIVE_FRACTION,
        rpn_box_coder,
    )


def make_roi_box_loss_evaluator(cfg):
    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)
    return make_standard_loss_evaluator(
        "fast_rcnn",
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
        cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION,
        box_coder,
    )


def make_roi_mask_loss_evaluator(cfg):
    return make_standard_loss_evaluator(
        "mask_rcnn",
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        mask_resolution=cfg.MODEL.ROI_MASK_HEAD.RESOLUTION,
        mask_subsample_only_positive_boxes=(not cfg.MODEL.ROI_HEADS.USE_FPN),
    )
