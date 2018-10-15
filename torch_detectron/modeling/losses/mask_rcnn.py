import torch
from torch.nn import functional as F

from ..utils import cat
from ..utils import keep_only_positive_boxes
from ..utils import nonzero
from ..utils import smooth_l1_loss
from .matcher import Matcher
from .target_preparator import TargetPreparator


class MaskTargetPreparator(TargetPreparator):
    """
    This class aligns the ground-truth targets to the anchors that are
    passed to the image.
    """

    def __init__(self, proposal_matcher, discretization_size):
        super(MaskTargetPreparator, self).__init__(proposal_matcher, None)
        self.discretization_size = discretization_size

    def index_target(self, target, index):
        """
        This function is used to index the targets, possibly only propagating a few
        fields of target (instead of them all). In this case, we only propagate
        labels and masks.

        Arguments:
            target (BoxList): an arbitrary bbox object, containing many possible fields
            index (Tensor): the indices to select.
        """

        target = target.copy_with_fields(["labels", "masks"])
        return target[index]

    def prepare_labels(self, matched_targets_per_image, anchors_per_image):
        """
        Arguments:
            matched_targets_per_image (BoxList): a BoxList with the 'matched_idx' field set,
                containing the ground-truth targets aligned to the anchors,
                i.e., it contains the same number of elements as the number of anchors,
                and contains de best-matching ground-truth target element. This is
                returned by match_targets_to_anchors
            anchors_per_image (a BoxList object)

        This method should return a single tensor, containing the labels
        for each element in the anchors
        """
        matched_idxs = matched_targets_per_image.get_field("matched_idxs")
        labels_per_image = matched_targets_per_image.get_field("labels")
        labels_per_image = labels_per_image.to(dtype=torch.int64)

        # this can probably be removed, but is left here for clarity
        # and completeness
        neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
        labels_per_image[neg_inds] = 0

        # mask scores are only computed on positive samples
        positive_inds = nonzero(labels_per_image > 0)[0]

        segmentation_masks = matched_targets_per_image.get_field("masks")
        segmentation_masks = segmentation_masks[positive_inds]

        positive_anchors = anchors_per_image[positive_inds]

        masks_per_image = self.project(segmentation_masks, positive_anchors)
        return masks_per_image, labels_per_image

    def project(self, segmentation_masks, positive_anchors):
        """
        Given segmentation masks and the bounding boxes corresponding
        to the location of the masks in the image, this function
        crops and resizes the masks in the position defined by the
        boxes. This prepares the masks for them to be fed to the
        loss computation as the targets.

        Arguments:
            segmentation_masks: an instance of SegmentationMask
            positive_anchors: an instance of BoxList
        """
        masks = []
        M = self.discretization_size
        device = positive_anchors.bbox.device
        positive_anchors = positive_anchors.convert("xyxy")
        assert segmentation_masks.size == positive_anchors.size, "{}, {}".format(
            segmentation_masks, positive_anchors
        )
        # TODO put the anchors on the CPU, as the representation for the
        # masks is not efficient GPU-wise (possibly several small tensors for
        # representing a single instance mask)
        positive_anchors = positive_anchors.bbox.to(torch.device("cpu"))
        for segmentation_mask, anchor in zip(segmentation_masks, positive_anchors):
            # crop the masks, resize them to the desired resolution and
            # then convert them to the tensor representation,
            # instead of the list representation that was used
            cropped_mask = segmentation_mask.crop(anchor)
            scaled_mask = cropped_mask.resize((M, M))
            mask = scaled_mask.convert(mode="mask")
            masks.append(mask)
        if len(masks) == 0:
            return torch.empty(0, dtype=torch.float32, device=device)
        return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


class MaskRCNNLossComputation(object):
    def __init__(self, target_preparator, subsample_only_positive_boxes=False):
        """
        If subsample_only_positive_boxes is False, all the boxes from the RPN
        that were passed to the detection branch will be used for mask loss
        computation. This is wasteful, as only the positive boxes are used
        for mask loss (which corresponds to 25% of a batch).
        If subsample_only_positive_boxes is True, then only the positive
        boxes are selected, but this only works with FPN-like architectures.
        """
        self.target_preparator = target_preparator
        self.subsample_only_positive_boxes = subsample_only_positive_boxes

    def prepare_targets(self, anchors, targets):
        """
        This reimplents parts of the functionality of TargetPreparator.__call__
        The reason being that we don't need bbox regression targets for
        masks, so I decided to specialize it here instead of modifying
        TargetPreparator. It might be worth considering modifying this once
        I implement keypoints
        """
        target_preparator = self.target_preparator
        # TODO assert / resize anchors to have the same .size as targets?
        matched_targets = target_preparator.match_targets_to_anchors(anchors, targets)
        labels = []
        masks = []
        for matched_targets_per_image, anchors_per_image in zip(
            matched_targets, anchors
        ):
            masks_per_image, labels_per_image = target_preparator.prepare_labels(
                matched_targets_per_image, anchors_per_image
            )
            labels.append(labels_per_image)
            masks.append(masks_per_image)
        return labels, masks

    def __call__(self, anchors, mask_logits, targets):
        """
        Arguments:
            anchors (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        if self.subsample_only_positive_boxes:
            anchors = keep_only_positive_boxes(anchors)
        labels, mask_targets = self.prepare_targets(anchors, targets)

        labels = cat(labels, dim=0)
        mask_targets = cat(mask_targets, dim=0)

        positive_inds = nonzero(labels > 0)[0]
        labels_pos = labels[positive_inds]

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0

        mask_loss = F.binary_cross_entropy_with_logits(
            mask_logits[positive_inds, labels_pos], mask_targets
        )
        return mask_loss
