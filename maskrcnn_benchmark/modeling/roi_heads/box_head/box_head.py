import torch
from torch.nn import functional as F

from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    sample_with_positive_fraction
)
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.layers import smooth_l1_loss

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor


def fastrcnn_losses(labels, regression_targets, class_logits, regression_outputs):
    """
    Computes the box classification & regression loss for Faster R-CNN.

    Arguments:
        labels (Tensor): #box binary labels.
        class_logits (Tensor): #box x #class
        box_regression (Tensor): #box x (#class x 4)

    Returns:
        classification_loss, regression_loss
    """
    device = class_logits.device

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)

    box_loss = smooth_l1_loss(
        regression_outputs[sampled_pos_inds_subset[:, None], map_inds],
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

        self.batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION

    def forward(self, features, proposals, targets=None):
        if self.training:
            with torch.no_grad():
                proposals, labels, regression_targets = self._prepare_regression_targets(proposals, targets)
                # #img BoxList

        x = self.feature_extractor(features, proposals)
        class_logits, regression_outputs = self.predictor(x)
        # #box x #class, #box x #class x 4

        if not self.training:
            result = self.post_processor((class_logits, regression_outputs), proposals)
            return x, result, {}
        else:
            loss_classifier, loss_box_reg = fastrcnn_losses(
                cat(labels, dim=0),
                cat(regression_targets, dim=0),
                class_logits, regression_outputs)
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
            list[BoxList]: The proposals after sampling.
            list[Tensor]: #img labels, each is a 1D tensor containing the label (0, positive) for all proposals on the image.
            list[Tensor]: #img regression_targets. Each has shape Nx4, the regression targets for all proposals on the image.
                Only those proposals with positive labels contains meaningful target values.
        """
        labels, regression_targets = [], []
        for image_idx, (proposals_per_image, targets_per_image) in enumerate(zip(proposals, targets)):
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

            # Label foreground
            labels_per_image = labels_per_image.clamp(max=1)
            # Label background (below the low threshold)
            labels_per_image[matched_idxs == Matcher.BELOW_LOW_THRESHOLD] = 0
            # Label ignore proposals (between low and high thresholds)
            labels_per_image[matched_idxs == Matcher.BETWEEN_THRESHOLDS] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            # apply sampling
            sampled_pos_inds, sampled_neg_inds = sample_with_positive_fraction(
                labels_per_image, self.batch_size_per_image, self.positive_sample_fraction)
            sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

            labels.append(labels_per_image[sampled_inds])
            regression_targets.append(regression_targets_per_image[sampled_inds])

            proposals[image_idx] = proposals_per_image[sampled_inds]
            proposals[image_idx].add_field("labels", labels[-1])

        return proposals, labels, regression_targets


def build_roi_box_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg)
