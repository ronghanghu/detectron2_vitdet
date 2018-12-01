"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist

from ..balanced_positive_negative_sampler import sample_with_positive_fraction
from ..utils import cat


class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, box_coder, batch_per_image, positive_fraction):
        """
        Arguments:
            proposal_matcher (Matcher)
            box_coder (BoxCoder)
            batch_per_image (int)
            positive_fraction (float)
        """
        self.proposal_matcher = proposal_matcher
        self.box_coder = box_coder
        self.batch_per_image = batch_per_image
        self.positive_fraction = positive_fraction

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            match_quality_matrix = boxlist_iou(targets_per_image, anchors_per_image)
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            # NB: need to clamp the indices because matched_idxs can be <0
            matched_targets = targets_per_image.bbox[matched_idxs.clamp(min=0)]

            labels_per_image = (matched_idxs >= 0).to(dtype=torch.float32)
            # discard anchors that go out of the boundaries of the image
            labels_per_image[~anchors_per_image.get_field("visibility")] = -1
            # discard indices that are neither foreground or background
            labels_per_image[matched_idxs == Matcher.BETWEEN_THRESHOLDS] = -1

            # compute regression targets
            # TODO wasted computation for ignored boxes
            regression_targets_per_image = self.box_coder.encode(
                matched_targets, anchors_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[list[BoxList]]): #img x #lvl BoxList
            objectness (list[Tensor]): #lvl tensors
            box_regression (list[Tensor]): #lvl tensors
            targets (list[BoxList]): #img BoxList
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]  # #img BoxLists
        labels, regression_targets = self.prepare_targets(anchors, targets)
        # labels: #img Tensors, each is NAx1
        # regression_targets: #img Tensors, each is NAx4

        sampled_pos_inds, sampled_neg_inds = [], []
        for label in labels:
            pos_idx, neg_idx = sample_with_positive_fraction(
                label, self.batch_per_image, self.positive_fraction
            )
            zeros = torch.zeros_like(label, dtype=torch.uint8)
            sampled_pos_inds.append(zeros.scatter(0, pos_idx, 1))
            sampled_neg_inds.append(zeros.scatter(0, neg_idx, 1))
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness_flattened = []
        box_regression_flattened = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the objectness and the box_regression
        for objectness_per_level, box_regression_per_level in zip(objectness, box_regression):
            N, A, H, W = objectness_per_level.shape
            objectness_per_level = objectness_per_level.permute(0, 2, 3, 1).reshape(N, -1)
            box_regression_per_level = box_regression_per_level.view(N, -1, 4, H, W)
            box_regression_per_level = box_regression_per_level.permute(0, 3, 4, 1, 2)
            box_regression_per_level = box_regression_per_level.reshape(N, -1, 4)
            objectness_flattened.append(objectness_per_level)
            box_regression_flattened.append(box_regression_per_level)
        # concatenate on the first dimension (representing the feature levels), to
        # take into account the way the labels were generated (with all feature maps
        # being concatenated as well)
        objectness = cat(objectness_flattened, dim=1).reshape(-1)
        box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss


def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )

    loss_evaluator = RPNLossComputation(
        matcher, box_coder, cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )
    return loss_evaluator
