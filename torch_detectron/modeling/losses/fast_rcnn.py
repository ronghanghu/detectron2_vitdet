import torch
from torch.nn import functional as F

from ..utils import cat
from ..utils import nonzero
from ..utils import smooth_l1_loss
from .matcher import Matcher
from .target_preparator import TargetPreparator


class FastRCNNTargetPreparator(TargetPreparator):
    """
    This class returns labels and regression targets for Fast R-CNN
    """

    def index_target(self, target, index):
        target = target.copy_with_fields("labels")
        return target[index]

    def prepare_labels(self, matched_targets_per_image, anchors_per_image):
        matched_idxs = matched_targets_per_image.get_field("matched_idxs")
        labels_per_image = matched_targets_per_image.get_field("labels")
        labels_per_image = labels_per_image.to(dtype=torch.int64)

        # Label background (below the low threshold)
        bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
        labels_per_image[bg_inds] = 0

        # Label ignore proposals (between low and high thresholds)
        ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
        labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler
        return labels_per_image


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(self, target_preparator, fg_bg_sampler):
        """
        Arguments:
            target_preparator: an instance of TargetPreparator
            fg_bg_sampler: an instance of BalancedPositiveNegativeSampler
        """
        self.target_preparator = target_preparator
        self.fg_bg_sampler = fg_bg_sampler

    def subsample(self, anchors, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled anchors.
        Note: this function keeps a state.

        Arguments:
            anchors (list[BoxList])
            targets (list[BoxList])
        """

        labels, regression_targets = self.target_preparator(anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        anchors = list(anchors)
        # add corresponding label information to the bounding boxes
        # this can be used with `keep_only_positive_boxes` in order to
        # restrict the set of boxes to be used during other steps (Mask R-CNN
        # for example)
        for labels_per_image, regression_targets_per_image, anchors_per_image in zip(labels, regression_targets, anchors):
            anchors_per_image.add_field("labels", labels_per_image)
            anchors_per_image.add_field("regression_targets", regression_targets_per_image)

        # distributed sampled anchors, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = nonzero(pos_inds_img | neg_inds_img)[0]
            anchors_per_image = anchors[img_idx][img_sampled_inds]
            anchors[img_idx] = anchors_per_image

        self._anchors = anchors
        return anchors

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_anchors"):
            raise RuntimeError("subsample needs to be called before")

        anchors = self._anchors

        labels = cat([anchor.get_field("labels") for anchor in anchors], dim=0)
        regression_targets = cat([anchor.get_field("regression_targets") for anchor in anchors], dim=0)

        classification_loss = F.cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = nonzero(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        ) / labels.numel()

        return classification_loss, box_loss
