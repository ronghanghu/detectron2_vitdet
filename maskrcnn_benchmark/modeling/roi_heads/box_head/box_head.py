import torch
from torch.nn import functional as F

from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.layers import smooth_l1_loss

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor


def fastrcnn_box_loss(proposals, class_logits, box_regression):
    """
    Computes the box classification & regression loss for Faster R-CNN.

    Arguments:
        class_logits (Tensor): #box x #class
        box_regression (Tensor): #box x (#class x 4)

    Returns:
        classification_loss, regression_loss
    """
    device = class_logits.device

    labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
    regression_targets = cat(
        [proposal.get_field("regression_targets") for proposal in proposals], dim=0
    )

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)

    box_loss = smooth_l1_loss(
        box_regression[sampled_pos_inds_subset[:, None], map_inds],
        regression_targets[sampled_pos_inds_subset],
        size_average=False,
        beta=1,
    )
    box_loss = box_loss / labels.numel()
    return classification_loss, box_loss


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        self.predictor = make_roi_box_predictor(cfg)
        self.post_processor = make_roi_box_post_processor(cfg)

        # match proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
            cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
            allow_low_quality_matches=False,
        )
        bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
        self.box_coder = BoxCoder(weights=bbox_reg_weights)

        self.fg_bg_sampler = BalancedPositiveNegativeSampler(
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        )

    def forward(self, features, proposals, targets=None):
        if self.training:
            with torch.no_grad():
                proposals = self._prepare_regression_targets(proposals, targets)
                # #img BoxList

        x = self.feature_extractor(features, proposals)
        class_logits, box_regression = self.predictor(x)
        # #box x #class, #box x #class x 4

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}
        else:
            loss_classifier, loss_box_reg = fastrcnn_box_loss(
                proposals, class_logits, box_regression)
            return (
                x,
                proposals,
                dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
            )

    def _prepare_regression_targets(self, proposals, targets):
        """
        Sample the proposals and prepare their training targets.

        Args:
            proposals (list[BoxList]): #img BoxList. Each contains the proposals for the image.
            targets (list[BoxList]): #img BoxList. The GT boxes for the image.

        Returns:
            The proposals after sampling, with two additional fields:
            labels (list[Tensor]): Each is a 1D tensor containing the label (-1, 0, 1) for each proposal.
            regression_targets: (list[Tensor]):  Each has shape Nx4, the regression targets for the proposals.
                Only those proposals with positive labels contains meaningful target values.
        """
        labels = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            match_quality_matrix = boxlist_iou(targets_per_image, proposals_per_image)
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            # Fast RCNN only need "labels" field for selecting the targets
            target = targets_per_image.copy_with_fields("labels")
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
            labels.append(labels_per_image)

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field("regression_targets", regression_targets_per_image)

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        proposals = [
            p[torch.nonzero(pos_inds | neg_inds).squeeze(1)]
            for p, pos_inds, neg_inds in zip(proposals, sampled_pos_inds, sampled_neg_inds)
        ]

        return proposals


def build_roi_box_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg)
