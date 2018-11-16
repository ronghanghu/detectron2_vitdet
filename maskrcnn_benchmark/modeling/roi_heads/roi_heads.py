import torch

from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.bounding_box import BoxList
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
    positive_inds = []
    positive_targets = []
    for boxes_per_image, targets_per_image in zip(boxes, matched_targets):
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_targets.append(targets_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_targets, positive_inds


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
        x, detections, losses = self.box(features, proposals, targets)

        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            if self.training:
                proposals, targets, pos_inds = keep_only_positive_boxes(proposals, targets)
                if self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                    # if sharing, don't need to do feature extraction again,
                    # just use the box features for all the positivie proposals
                    mask_features = x[torch.cat(pos_inds, dim=0)]

            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)
        return x, detections, losses


def build_roi_heads(cfg):
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
