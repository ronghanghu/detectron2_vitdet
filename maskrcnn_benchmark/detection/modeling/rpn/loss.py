"""
This file contains specific functions for computing losses on the RPN
file
"""

import numpy as np
import torch
from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, cat_boxlist
from torch.nn import functional as F

from ..balanced_positive_negative_sampler import sample_with_positive_fraction
from ..matcher import Matcher


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

            labels_per_image = (matched_idxs >= 0).to(dtype=torch.int32)
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
            objectness (list[Tensor]): #lvl tensors, each tensor has shape (N, #anchors, H, W)
            box_regression (list[Tensor]): #lvl tensors, each tensor has shape (N, #anchorsx4, H, W)
            targets (list[BoxList]): #img BoxList
        """
        num_levels = len(objectness)
        num_images = len(targets)
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]  # #img BoxLists
        labels, regression_targets = self.prepare_targets(anchors, targets)
        # labels: #img Tensors, each is a vector whose length is the total number
        #   of all anchors (#all_anchors). Values in {-1, 0, 1}
        # regression_targets: #img Tensors, each is #all_anchors x4

        def resample(label):
            pos_idx, neg_idx = sample_with_positive_fraction(
                label, self.batch_per_image, self.positive_fraction
            )
            label.fill_(-1)
            label.scatter_(0, pos_idx, 1)
            label.scatter_(0, neg_idx, 0)
            return label

        labels = torch.stack([resample(label) for label in labels], dim=0)  # N x #all_anchors
        num_anchors_per_level = [np.prod(x.shape[1:]) for x in objectness]
        total_num_anchors = sum(num_anchors_per_level)
        assert labels.shape[1] == total_num_anchors
        labels = torch.split(labels, num_anchors_per_level, dim=1)  # N x (HxWx#anchors_per_level)

        regression_targets = torch.stack(regression_targets, dim=0)
        assert regression_targets.shape[1] == total_num_anchors
        regression_targets = torch.split(regression_targets, num_anchors_per_level, dim=1)

        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        obj_losses, box_losses = [], []
        for lvl in range(num_levels):
            labels_per_level, objectness_per_level, box_regression_per_level, regression_targets_per_level = (
                labels[lvl],
                objectness[lvl],
                box_regression[lvl],
                regression_targets[lvl],
            )
            N, A, H, W = objectness_per_level.shape
            objectness_per_level = objectness_per_level.permute(0, 2, 3, 1).flatten()
            box_regression_per_level = box_regression_per_level.view(N, -1, 4, H, W)
            box_regression_per_level = box_regression_per_level.permute(0, 3, 4, 1, 2).reshape(
                -1, 4
            )

            obj_loss, box_loss = rpn_losses(
                labels_per_level.flatten(),
                regression_targets_per_level.reshape(-1, 4),
                objectness_per_level,
                box_regression_per_level,
            )
            obj_losses.append(obj_loss)
            box_losses.append(box_loss)

        obj_loss, box_loss = sum(obj_losses), sum(box_losses)
        normalizer = self.batch_per_image * num_images
        return obj_loss / normalizer, box_loss / normalizer


def rpn_losses(labels, regression_targets, label_logits, regression_predictions):
    """
    Args:
        labels: (N,), each in {-1, 0, 1}. -1 means ignore
        regression_targets: (N, 4)
        label_logits: (N,)
        regression_predictions: (N, 4)
    Returns:
        objectness_loss, box_loss, both unnormalized (summed over samples).
    """
    pos_masks = labels == 1
    box_loss = smooth_l1_loss(
        regression_predictions[pos_masks],
        regression_targets[pos_masks],
        beta=1.0 / 9,
        size_average=False,
    )

    valid_masks = labels >= 0
    objectness_loss = F.binary_cross_entropy_with_logits(
        label_logits[valid_masks], labels[valid_masks].to(torch.float32), reduction="sum"
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
