"""
Proposal implementation for the model builder
Everything that constructs something is a function that is
or can be accessed by a string
"""

import torch
from torch import nn

import torch_detectron.modeling as M
from torch_detectron.modeling import resnet


class GeneralizedRCNN(nn.Module):
    """
    Main class for generalized r-cnn. Supports boxes, masks and keypoints
    This is very similar to what we had before, the difference being that now
    we construct the modules in __init__, instead of passing them as arguments
    """
    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.cfg = cfg.copy()
        
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
        self.roi_heads = combine_roi_heads(cfg, roi_heads)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BBox]): ground-truth boxes present in the image (optional)
        """
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.cfg.RPN_ONLY:
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

################################################################################
# Those generic classes shows that it's possible to plug both Mask R-CNN C4
# and Mask R-CNN FPN with generic containers
################################################################################
class ParallelHead(torch.nn.Module):
    """
    For Mask R-CNN FPN
    """
    def __init__(self, heads):
        super(ParallelHead, self).__init__()
        self.heads = torch.nn.ModuleList(heads)

    def forward(self, features, proposals, targets=None):
        losses = {}
        for head in self.heads:
            x, proposals, loss = head(features, proposals, targets)
            losses.update(loss)

        return x, proposals, losses


class SequentialHead(torch.nn.Module):
    """
    For Mask R-CNN C4
    """
    def __init__(self, heads):
        super(SequentialHead, self).__init__()
        self.heads = torch.nn.ModuleList(heads)

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
    constructor = ParallelHead
    if cfg.MODEL.SHARE_FEATURES_DURING_TRAINING:
        constructor = SequentialHead
    # can also use getfunc to query a function by name
    return constructor(roi_heads)


################################################################################
# Create RPN
################################################################################
# copied from the previous implementation
def make_anchor_generator(config):
    use_fpn = config.MODEL.RPN.USE_FPN

    scales = config.MODEL.RPN.SCALES
    aspect_ratios = config.MODEL.RPN.ASPECT_RATIOS
    base_anchor_size = config.MODEL.RPN.BASE_ANCHOR_SIZE
    anchor_stride = config.MODEL.RPN.ANCHOR_STRIDE
    straddle_thresh = config.MODEL.RPN.STRADDLE_THRESH

    anchor_maker = M.AnchorGenerator if not use_fpn else M.FPNAnchorGenerator
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
    return anchor_generator


# copied from the previous implementation
def make_box_selector(config, rpn_box_coder, is_train):
    use_fpn = config.MODEL.RPN.USE_FPN
    box_selector_maker = M.RPNBoxSelector
    box_selector_args = {}
    if use_fpn:
        # TODO expose those options
        roi_to_fpn_level_mapper = M.ROI2FPNLevelsMapper(2, 5)
        box_selector_maker = M.FPNRPNBoxSelector
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


def make_rpn(cfg):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    anchor_generator = make_anchor_generator(cfg)
    ...
    return rpn


################################################################################
# RoiBoxHead
################################################################################
def make_roi_box_feature_extractor(cfg):
    func = getfunc(cfg.MODEL.ROI_BOX.FEATURE_EXTRACTOR)
    # e.g., cfg.MODEL.ROI_BOX.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
    return func(cfg)

def make_roi_box_predictor(cfg):
    func = getfunc(cfg.MODEL.ROI_BOX.PREDICTOR)
    # e.g., cfg.MODEL.ROI_BOX.PREDICTOR = "FastRCNNPredictor"
    return func(cfg)


def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = M.BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG

    postprocessor_maker = M.PostProcessor if not use_fpn else M.FPNPostProcessor
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
        return x, proposals, dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)


def build_roi_box_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg)


###
# Those implementations are here for illustrative purposes, and they will be in separate files
# depending on the application
###

class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config, pretrained=None):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = M.Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            drop_last=False,
        )

        stage = resnet.StageSpec(index=5, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNET.BLOCK_MODULE,
            stages=(stage,),
            num_groups=config.MODEL.RESNET.NUM_GROUPS,
            width_per_group=config.MODEL.RESNET.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNET.STRIDE_IN_1X1,
            stride_init=1 if "QUICK_SCHEDULE" in os.environ and os.environ["QUICK_SCHEDULE"] else None,
        )

        if pretrained:
            state_dict = torch.load(pretrained)
            load_state_dict(head, state_dict, strict=False)
        
        self.pooler = pooler
        self.head = head

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


class FastRCNNPredictor(nn.Module):
    def __init__(self, config, pretrained=None):
        super(FastRCNNPredictor, self).__init__()
        num_inputs = 2048  # config.MODEL.ROI_BOX_HEAD.something
        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = torch.nn.AvgPool2d(kernel_size=7, stride=7)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.bbox_pred = nn.Linear(num_inputs, num_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.weight, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.weight, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred



################################################################################
# RoiMaskHead
################################################################################
class ROIMaskHead(torch.nn.Module):
    def __init__(self, feature_extractor, predictor, post_processor, loss_evaluator):
        super(ROIMaskHead, self).__init__()
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

        x = self.feature_extractor(features, proposals)
        mask_logits = self.predictor(x)

        if not self.training:
            result = self.post_processor(mask_logits, proposals)
            return x, result, {}

        loss_mask = self.loss_evaluator(proposals, mask_logits, targets)

        return x, proposals, dict(loss_mask=loss_mask)


def build_roi_mask_head(cfg):
    return ROIMaskHead(cfg)


# the rest follows a similar pattern as the one from RoiBoxHead

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
    matcher = M.Matcher(
        fg_iou_threshold,
        bg_iou_threshold,
        allow_low_quality_matches=allow_low_quality_matches,
    )

    if loss_type in ("rpn", "fast_rcnn"):
        assert isinstance(batch_size_per_image, int)
        assert isinstance(positive_fraction, (int, float))
        assert isinstance(box_coder, BoxCoder)
        fg_bg_sampler = M.BalancedPositiveNegativeSampler(
            batch_size_per_image=batch_size_per_image,
            positive_fraction=positive_fraction,
        )

    if loss_type == "rpn":
        for arg in (mask_resolution, mask_subsample_only_positive_boxes):
            assert arg is None
        target_preparator = M.RPNTargetPreparator(matcher, box_coder)
        loss_evaluator = M.RPNLossComputation(target_preparator, fg_bg_sampler)
    elif loss_type == "fast_rcnn":
        for arg in (mask_resolution, mask_subsample_only_positive_boxes):
            assert arg is None
        target_preparator = M.FastRCNNTargetPreparator(matcher, box_coder)
        loss_evaluator = M.FastRCNNLossComputation(target_preparator, fg_bg_sampler)
    elif loss_type == "mask_rcnn":
        for arg in (batch_size_per_image, positive_fraction):
            assert arg is None
        assert isinstance(mask_resolution, (int, float))
        assert isinstance(mask_subsample_only_positive_boxes, bool)
        target_preparator = M.MaskTargetPreparator(matcher, mask_resolution)
        loss_evaluator = M.MaskRCNNLossComputation(
            target_preparator,
            subsample_only_positive_boxes=mask_subsample_only_positive_boxes,
        )

    return loss_evaluator


def make_rpn_loss_evaluator(cfg):
    rpn_box_coder = M.BoxCoder(weights=(1., 1., 1., 1.))
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
    box_coder = M.BoxCoder(weights=bbox_reg_weights)
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
        mask_resolution=cfg.MODEL.ROI_HEADS.MASK_RESOLUTION,
        mask_subsample_only_positive_boxes=(not cfg.MODEL.USE_FPN),
    )
