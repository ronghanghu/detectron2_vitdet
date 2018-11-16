import torch

from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    sample_with_positive_fraction
)

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head


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
                proposals, matched_targets, labels = self._sample_proposals_for_training(proposals, targets)
                assert len(labels) == len(proposals)
                assert len(proposals) == len(matched_targets)
                # #img BoxList
        else:
            matched_targets = None

        losses = {}


        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x, detections, loss_box = self.box(features, proposals, labels, matched_targets)
        losses.update(loss_box)
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_mask = self.mask(mask_features, detections, matched_targets)
            losses.update(loss_mask)
        return x, detections, losses

    def _sample_proposals_for_training(self, proposals, targets):
        """
        Sample the proposals and prepare their training targets.

        Args:
            proposals (list[BoxList]): #img BoxList. Each contains the proposals for the image.
            targets (list[BoxList]): #img BoxList. The GT boxes for the image.

        Returns:
        # TODO
            list[BoxList]: The proposals after sampling.
            list[Tensor]: #img labels, each is a 1D tensor containing the label (0, positive) for all proposals on the image.
            list[Tensor]: #img regression_targets. Each has shape Nx4, the regression targets for all proposals on the image.
                Only those proposals with positive labels contains meaningful target values.

# TODO return sampled_proposals and matched_targets instead.
        """
        labels, sampled_targets = [], []

        for image_idx, (proposals_per_image, targets_per_image) in enumerate(zip(proposals, targets)):
            match_quality_matrix = boxlist_iou(targets_per_image, proposals_per_image)
            matched_idxs = self.proposal_matcher(match_quality_matrix)
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
                labels_per_image, self.batch_size_per_image, self.positive_sample_fraction)
            sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

            labels.append(labels_per_image[sampled_inds])
            sampled_targets.append(matched_targets[sampled_inds])

            proposals[image_idx] = proposals_per_image[sampled_inds]
            # TODO avoid the use of field here
            proposals[image_idx].add_field("labels", labels[-1])

        return proposals, sampled_targets, labels

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
