import torch
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    sample_with_positive_fraction
)

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head


def sample_proposals_for_training(
        proposals, targets,
        proposal_matcher, batch_size_per_image, positive_sample_fraction):
    """
    Sample the proposals and prepare their training targets.

    Args:
        proposals (list[BoxList]): #img BoxList. Each contains the proposals for the image.
        targets (list[BoxList]): #img BoxList. The GT boxes for the image.
        proposal_matcher (Matcher):
        batch_size_per_image (int)
        positive_sample_fraction (float)

    Returns:
        list[BoxList]: The proposals after sampling. It has a "labels" field, ranging in [0, #class]
        list[BoxList]: The matched targets for each sampled proposal.
            Only those for foreground proposals are meaningful.
    """
    sampled_targets = []

    for image_idx, (proposals_per_image, targets_per_image) in enumerate(zip(proposals, targets)):
        match_quality_matrix = boxlist_iou(targets_per_image, proposals_per_image)
        matched_idxs = proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = targets_per_image.copy_with_fields(["labels", "masks"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]

        labels_per_image = matched_targets.get_field("labels").to(dtype=torch.int64)

        # Label background (below the low threshold)
        labels_per_image[matched_idxs == Matcher.BELOW_LOW_THRESHOLD] = 0
        # Label ignore proposals (between low and high thresholds)
        labels_per_image[matched_idxs == Matcher.BETWEEN_THRESHOLDS] = -1  # -1 is ignored by sampler

        # apply sampling
        sampled_pos_inds, sampled_neg_inds = sample_with_positive_fraction(
            labels_per_image, batch_size_per_image, positive_sample_fraction)
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        sampled_targets.append(matched_targets[sampled_inds])
        proposals[image_idx] = proposals_per_image[sampled_inds]
        proposals[image_idx].add_field("labels", labels_per_image[sampled_inds])
    return proposals, sampled_targets


def keep_only_positive_boxes(boxes, matched_targets):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list[BoxList])
        matched_targets (list[BoxList]):
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_masks = []
    positive_targets = []
    for boxes_per_image, targets_per_image in zip(boxes, matched_targets):
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_targets.append(targets_per_image[inds])
        positive_masks.append(inds_mask)
    return positive_boxes, positive_targets, positive_masks


class CombinedROIHeads(torch.nn.ModuleDict):
    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor

        # match proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
            cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
            allow_low_quality_matches=False,
        )
        self.batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION

    def forward(self, features, proposals, targets=None):
        if self.training:
            with torch.no_grad():
                proposals, targets = sample_proposals_for_training(
                    proposals, targets, self.proposal_matcher, self.batch_size_per_image, self.positive_sample_fraction)
                assert len(proposals) == len(targets)
                # #img BoxList
        else:
            targets = None

        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x, proposals, losses = self.box(features, proposals, targets)

        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            if self.training:
                proposals, targets, pos_masks = keep_only_positive_boxes(proposals, targets)
                if self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                    # if sharing, don't need to do feature extraction again,
                    # just use the box features for all the positivie proposals
                    mask_features = x[torch.cat(pos_masks, dim=0)]

            # During training, self.box() will return the unaltered proposals as "proposals"
            # this makes the API consistent during training and testing
            x, proposals, loss_mask = self.mask(mask_features, proposals, targets)
            losses.update(loss_mask)
        return x, proposals, losses


class ROIHeads(torch.nn.Module):
    def __init__(self, cfg):
        super(ROIHeads, self).__init__()
        self.cfg = cfg.clone()

        # match proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
            cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
            allow_low_quality_matches=False,
        )
        self.batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION

        self.test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
        self.test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS
        self.test_detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG

    def forward(self, features, proposals, targets=None):
        if self.training:
            with torch.no_grad():
                proposals, targets = sample_proposals_for_training(
                    proposals, targets, self.proposal_matcher, self.batch_size_per_image, self.positive_sample_fraction)
            return self.forward_training(features, proposals, targets)
        else:
            return self.forward_inference(features, proposals)

    def fastrcnn_predictions(self, outputs):
        """
        Args:
            outputs (FastRCNNOutputs):

        Returns:
            list[BoxList]: the predictions for each image. Contains field "labels" and "scores".
        """
        decoded_boxes = outputs.decoded_outputs()
        probs = outputs.predicted_probs()

        results = []
        for probs_per_image, boxes_per_image, image_shape in zip(probs, decoded_boxes, outputs.img_shapes):
            boxlist = fastrcnn_inference(
                boxes_per_image, probs_per_image, image_shape,
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img)
            results.append(boxlist)
        return results

    def forward_training(self, features, proposals, targets, labels):
        """
        TODO
        """
        raise NotImplementedError()

    def forward_inference(self, features, proposals):
        """
        TODO
        """
        raise NotImplementedError()

    def forward_box_training():
        pass


from .box_head.roi_box_predictors import BoxHeadPredictor
from .box_head.box_head import FastRCNNOutputs, fastrcnn_inference
from .mask_head.mask_head import maskrcnn_loss
from .mask_head.inference import make_roi_mask_post_processor


class C4ROIHeads(ROIHeads):
    def __init__(self, cfg):
        super(C4ROIHeads, self).__init__(cfg)

        from .box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
        from .mask_head.roi_mask_predictors import MaskRCNNC4Predictor

        self.shared_roi_transform = ResNet50Conv5ROIFeatureExtractor(cfg)

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_inputs = res2_out_channels * stage2_relative_factor
        self.box_predictor = BoxHeadPredictor(cfg, num_inputs)

        self.box_coder = BoxCoder(weights=cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS)

        if cfg.MODEL.MASK_ON:
            self.mask_head = MaskRCNNC4Predictor(cfg)  # just 1 deconv
            self.mask_discretization_size = cfg.MODEL.ROI_MASK_HEAD.RESOLUTION
            self.mask_post_processor = make_roi_mask_post_processor()  # TODO make it a function

    def forward_training(self, features, proposals, targets):
        features = self.shared_roi_transform(features, proposals)

        feature_pooled = F.avg_pool2d(features, 7)  # gap
        class_logits, regression_outputs = self.box_predictor(feature_pooled)
        del feature_pooled

        losses = FastRCNNOutputs(
            self.box_coder, class_logits, regression_outputs, proposals, targets).losses()

        if self.cfg.MODEL.MASK_ON:
            proposals, targets, pos_masks = keep_only_positive_boxes(proposals, targets)
            # don't need to do feature extraction again,
            # just use the box features for all the positivie proposals
            mask_features = features[torch.cat(pos_masks, dim=0)]
            mask_logits = self.mask_head(mask_features)
            losses['loss_mask'] = maskrcnn_loss(
                proposals, mask_logits, targets, self.mask_discretization_size
            )
        return None, None, losses  # just to make outer API compatible

    def forward_inference(self, features, proposals):
        x = self.shared_roi_transform(features, proposals)
        x = F.avg_pool2d(x, 7)
        class_logits, regression_outputs = self.box_predictor(x)
        fastrcnn_outputs = FastRCNNOutputs(
            self.box_coder, class_logits, regression_outputs, proposals, None)
        results = self.fastrcnn_predictions(fastrcnn_outputs)

        if self.cfg.MODEL.MASK_ON:
            x = self.shared_roi_transform(features, results)
            mask_logits = self.mask_head(x)
            results = self.mask_post_processor(mask_logits, results)
        return None, results, {}  # just to make outer API compatible


def build_roi_heads(cfg):
    if not cfg.MODEL.ROI_HEADS.USE_FPN:   # TODO not a good if
        return C4ROIHeads(cfg)
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
