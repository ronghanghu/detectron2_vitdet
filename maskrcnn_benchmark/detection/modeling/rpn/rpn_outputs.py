import numpy as np
import torch
import torch.nn.functional as F

from maskrcnn_benchmark.layers import cat, smooth_l1_loss
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import (
    boxlist_iou,
    boxlist_nms,
    cat_boxlist,
    remove_small_boxes,
)

from ..matcher import Matcher
from ..sampling import subsample_labels


# TODO: comments for future refactoring of this module
#
# From @rbg:
# This code involves a significant amount of tensor reshaping and permuting. Look for
# ways to simplify this.

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    A: number of cell anchors (must be the same for all feature maps)
    Hi, Wi: height and width of the i-th feature map
    4: size of the box parameterization

Naming convention:

    objectness: refers to the binary classification of an anchor as object vs. not
    object.

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    objectness_logits_pred: predicted objectness scores in [-inf, +inf]; use
        sigmoid(objectness_logits_pred) to estimate P(object).

    objectness_logits_gt: ground-truth binary classification labels for objectness

    anchor_deltas_pred: predicted box2box transform deltas

    anchor_deltas_gt: ground-truth box2box transform deltas
"""


def sample_rpn_proposals(
    proposals,
    objectness_logits_pred,
    images,
    nms_thresh,
    pre_nms_topk,
    post_nms_topk,
    min_box_side_len,
    training,
):
    """
    Return scored object proposals after applying box regression to the anchors, NMS,
    taking the top k, etc.

    Args:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
        objectness_logits_pred (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
        images (ImageList): Input images as an :class:`ImageList`.
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is per
            feature map.
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.
        min_box_side_len (float): minimum proposal box side length in pixels (absolute units
            wrt input images).
        training (bool): True if proposals are to be used in training, otherwise False.
            This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..."
            comment.

    Returns:
        proposals (list[BoxLists]): list of N BoxLists. BoxList i stores post_nms_topk object
            proposals for image i.
    """
    # FIXME: make size order consistent everywhere: ImageList uses (height, width)
    # while BoxList uses (width, height)
    image_sizes = [size[::-1] for size in images.image_sizes]  # to (w, h) order

    proposals = [
        _sample_rpn_proposals_single_feature_map(
            proposals_i,
            objectness_logits_pred_i,
            image_sizes,
            nms_thresh,
            pre_nms_topk,
            post_nms_topk,
            min_box_side_len,
        )
        for proposals_i, objectness_logits_pred_i in zip(proposals, objectness_logits_pred)
    ]  # (L, N) BoxList.

    if len(proposals) == 1:
        # Single feature map, no need to merge
        return proposals[0]

    # Merge proposals from all levels within each image
    proposals = list(zip(*proposals))  # transpose to (N, L) BoxList
    # concat proposals across feature maps
    proposals = [cat_boxlist(boxlist) for boxlist in proposals]  # cat to (N, ) BoxList
    num_images = len(proposals)

    # NB: Legacy bug to address -- there's different behavior during training vs. testing.
    # During training, topk is over *all* the proposals combined over all images.
    # During testing, it is over the proposals for each image separately.
    # TODO: resolve this difference and make it consistent. It should be per image,
    # and not per batch, see: https://github.com/facebookresearch/Detectron/issues/459.
    if training:
        objectness_logits_pred = torch.cat(
            [boxlist.get_field("objectness_logits") for boxlist in proposals], dim=0
        )
        box_sizes = [len(boxlist) for boxlist in proposals]
        topk = min(post_nms_topk, len(objectness_logits_pred))
        _, inds_sorted = torch.topk(objectness_logits_pred, topk, dim=0, sorted=True)
        inds_mask = torch.zeros_like(objectness_logits_pred, dtype=torch.uint8)
        inds_mask[inds_sorted] = 1
        inds_mask = inds_mask.split(box_sizes)
        for i in range(num_images):
            proposals[i] = proposals[i][inds_mask[i]]
    else:
        for i in range(num_images):
            objectness_logits_pred = proposals[i].get_field("objectness_logits")
            topk = min(post_nms_topk, len(objectness_logits_pred))
            _, inds_sorted = torch.topk(objectness_logits_pred, topk, dim=0, sorted=True)
            proposals[i] = proposals[i][inds_sorted]
    return proposals


def _sample_rpn_proposals_single_feature_map(
    proposals,
    objectness_logits_pred,
    image_sizes,
    nms_thresh,
    pre_nms_topk,
    post_nms_topk,
    min_box_side_len,
):
    """
    Applies NMS, clips proposals, removes small proposals, and the returns the
    `post_nms_topk` highest scoring proposals for a single feature map.

    Args:
        image_sizes: N (width, height) tuples.
        Otherwise, see `sample_rpn_proposals`.

    Returns:
        list[BoxLists]: list of N BoxLists. BoxList i stores post_nms_topk object
            proposals for image i.
    """
    N, Hi_Wi_A = objectness_logits_pred.shape
    device = objectness_logits_pred.device
    assert proposals.shape[:2] == (N, Hi_Wi_A)

    pre_nms_topk = min(pre_nms_topk, Hi_Wi_A)
    objectness_logits_pred, topk_idx = objectness_logits_pred.topk(pre_nms_topk, dim=1)

    batch_idx = torch.arange(N, device=device)[:, None]
    proposals = proposals[batch_idx, topk_idx]  # shape is now (N, pre_nms_topk, 4)

    result = []
    for proposals_i, objectness_logits_pred_i, image_size_i in zip(
        proposals, objectness_logits_pred, image_sizes
    ):
        """
        proposals_i: tensor of shape (topk, 4)
        objectness_logits_pred_i: top-k objectness_logits_pred_i of shape (topk, )
        image_size_i: image (width, height)
        """
        boxlist = BoxList(proposals_i, image_size_i, mode="xyxy")
        # TODO why using "field" at all? Avoid storing arbitrary states
        boxlist.add_field("objectness_logits", objectness_logits_pred_i)
        boxlist = boxlist.clip_to_image(remove_empty=False)
        boxlist = remove_small_boxes(boxlist, min_box_side_len)
        boxlist = boxlist_nms(
            boxlist, nms_thresh, topk=post_nms_topk, score_field="objectness_logits"
        )
        result.append(boxlist)
    return result


def rpn_losses(objectness_logits_gt, anchor_deltas_gt, objectness_logits_pred, anchor_deltas_pred):
    """
    Args:
        objectness_logits_gt (Tensor): shape (N,), each element in {-1, 0, 1} representing
            ground-truth objectness labels with: -1 = ignore; 0 = not object; 1 = object.
        anchor_deltas_gt (Tensor): shape (N, 4), row i represents ground-truth
            box2box transform targets (dx, dy, dw, dh) that map anchor i to its
            matched ground-truth box.
        objectness_logits_pred (Tensor): shape (N,), each element is a predicted objectness
            logit.
        anchor_deltas_pred (Tensor): shape (N, 4), each row is a predicted box2box
            transform (dx, dy, dw, dh).

    Returns:
        objectness_loss, localization_loss, both unnormalized (summed over samples).
    """
    pos_masks = objectness_logits_gt == 1
    localization_loss = smooth_l1_loss(
        anchor_deltas_pred[pos_masks], anchor_deltas_gt[pos_masks], beta=1.0 / 9, size_average=False
    )

    valid_masks = objectness_logits_gt >= 0
    objectness_loss = F.binary_cross_entropy_with_logits(
        objectness_logits_pred[valid_masks],
        objectness_logits_gt[valid_masks].to(torch.float32),
        reduction="sum",
    )
    return objectness_loss, localization_loss


class RPNOutputs(object):
    def __init__(
        self,
        box2box_transform,
        anchor_matcher,
        batch_size_per_image,
        positive_fraction,
        images,
        objectness_logits_pred,
        anchor_deltas_pred,
        anchors,
        gt_box_list=None,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform): :class:`Box2BoxTransform` instance for
                anchor-proposal tranformations.
            anchor_matcher (Matcher): :class:`Matcher` instance for matching anchors to
                ground-truth boxes; used to determine training labels.
            batch_size_per_image (int): number of proposals to sample when training
            positive_fraction (float): target fraction of sampled proposals that should be positive
            images (ImageList): :class:`ImageList` instance representing N input images
            objectness_logits_pred (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, A, Hi, Wi) representing the predicted objectness logits for anchors.
            anchor_deltas_pred (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, A*4, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
            anchors (list[list[BoxLists]]): A list of N elements. Each element is a list of L
                BoxLists. BoxList (n, l) stores the entire anchor array for feature map l in image
                n (i.e. the cell anchors repeated over all locations in feature map (n, l)).
            gt_box_list (list[BoxList], optional): A list of N elements. Element i a BoxList storing
                the ground-truth ("gt") boxes for image i.
        """
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.objectness_logits_pred = objectness_logits_pred
        self.anchor_deltas_pred = anchor_deltas_pred
        self.anchors = anchors
        self.gt_box_list = gt_box_list
        self.num_feature_maps = len(objectness_logits_pred)
        self.num_images = len(images)

    def _get_ground_truth(self):
        """
        Returns:
            objectness_logits_gt: list of N tensors. Tensor i is a vector whose length is the
                total number of anchors in image i (i.e., len(anchors[i])). Label values are
                in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
            anchor_deltas_gt: list of N tensors. Tensor i has shape (len(anchors[i]), 4).
        """
        objectness_logits_gt = []
        anchor_deltas_gt = []
        # Concatenate anchors from all feature maps into a single BoxList per image
        anchors = [cat_boxlist(anchors_i) for anchors_i in self.anchors]
        for anchors_i, gt_box_list_i in zip(anchors, self.gt_box_list):
            """
            anchors_i: anchors for i-th image
            gt_box_list_i: ground-truth boxes for i-th image
            """
            match_quality_matrix = boxlist_iou(gt_box_list_i, anchors_i)
            matched_idxs = self.anchor_matcher(match_quality_matrix)
            # NB: need to clamp the indices because matched_idxs can be <0
            matched_gt_boxes = gt_box_list_i.bbox[matched_idxs.clamp(min=0)]

            objectness_logits_gt_i = (matched_idxs >= 0).to(dtype=torch.int32)
            # discard anchors that go out of the boundaries of the image
            objectness_logits_gt_i[~anchors_i.get_field("visibility")] = -1
            # discard indices that are neither foreground or background
            objectness_logits_gt_i[matched_idxs == Matcher.BETWEEN_THRESHOLDS] = -1

            # TODO wasted computation for ignored boxes
            anchor_deltas_gt_i = self.box2box_transform.get_deltas(anchors_i.bbox, matched_gt_boxes)

            objectness_logits_gt.append(objectness_logits_gt_i)
            anchor_deltas_gt.append(anchor_deltas_gt_i)

        return objectness_logits_gt, anchor_deltas_gt

    def losses(self):
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """

        def resample(label):
            """
            Randomly sample a subset of positive and negative examples by overwritting
            the label vector to the ignore value (-1) for all elements that are not
            included in the sample.
            """
            pos_idx, neg_idx = subsample_labels(
                label, self.batch_size_per_image, self.positive_fraction
            )
            # Fill with the ignore label (-1), then set positive and negative labels
            label.fill_(-1)
            label.scatter_(0, pos_idx, 1)
            label.scatter_(0, neg_idx, 0)
            return label

        objectness_logits_gt, anchor_deltas_gt = self._get_ground_truth()
        """
        objectness_logits_gt: list of N tensors. Tensor i is a vector whose length is the
            total number of anchors in image i (i.e., len(anchors[i]))
        anchor_deltas_gt: list of N tensors. Tensor i has shape (len(anchors[i]), 4).
        """

        # Collect all objectness labels and delta targets over feature maps and images
        # The final ordering is L, N, H, W, A from slowest to fastest axis.
        num_anchors_per_map = [np.prod(x.shape[1:]) for x in self.objectness_logits_pred]
        num_anchors_per_image = sum(num_anchors_per_map)

        # Stack to: (N, num_anchors_per_image)
        objectness_logits_gt = torch.stack(
            [resample(label) for label in objectness_logits_gt], dim=0
        )
        assert objectness_logits_gt.shape[1] == num_anchors_per_image
        # Split to tuple of L tensors, each with shape (N, num_anchors_per_map)
        objectness_logits_gt = torch.split(objectness_logits_gt, num_anchors_per_map, dim=1)
        # Concat from all feature maps
        objectness_logits_gt = cat([x.flatten() for x in objectness_logits_gt], dim=0)

        # Stack to: (N, num_anchors_per_image, 4)
        anchor_deltas_gt = torch.stack(anchor_deltas_gt, dim=0)
        assert anchor_deltas_gt.shape[1] == num_anchors_per_image
        # Split to tuple of L tensors, each with shape (N, num_anchors_per_image)
        anchor_deltas_gt = torch.split(anchor_deltas_gt, num_anchors_per_map, dim=1)
        # Concat from all feature maps
        anchor_deltas_gt = cat([x.reshape(-1, 4) for x in anchor_deltas_gt], dim=0)

        # Collect all objectness logits and delta predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W, A from slowest to fastest axis.
        objectness_logits_pred = cat(
            [
                # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N*Hi*Wi*A, )
                x.permute(0, 2, 3, 1).flatten()
                for x in self.objectness_logits_pred
            ],
            dim=0,
        )
        anchor_deltas_pred = cat(
            [
                # Reshape: (N, A*4, Hi, Wi) -> (N, A, 4, Hi, Wi) -> (N, Hi, Wi, A, 4)
                #          -> (N*Hi*Wi*A, 4)
                x.view(x.shape[0], -1, 4, x.shape[-2], x.shape[-1])
                .permute(0, 3, 4, 1, 2)
                .reshape(-1, 4)
                for x in self.anchor_deltas_pred
            ],
            dim=0,
        )

        objectness_loss, localization_loss = rpn_losses(
            objectness_logits_gt, anchor_deltas_gt, objectness_logits_pred, anchor_deltas_pred
        )
        normalizer = 1.0 / (self.batch_size_per_image * self.num_images)
        loss_cls = objectness_loss * normalizer  # cls: classification loss
        loss_loc = localization_loss * normalizer  # loc: localization loss
        losses = {"loss_rpn_cls": loss_cls, "loss_rpn_loc": loss_loc}

        return losses

    def predict_proposals(self):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, 4).
        """
        proposals = []
        # Transpose anchors from images-by-feature-maps (N, L) to feature-maps-by-images (L, N)
        anchors = list(zip(*self.anchors))
        # For each feature map
        for anchors_i, anchor_deltas_pred_i in zip(anchors, self.anchor_deltas_pred):
            N, _, Hi, Wi = anchor_deltas_pred_i.shape
            # Reshape: (N, A*4, Hi, Wi) -> (N, A, 4, Hi, Wi) -> (N, Hi, Wi, A, 4) -> (N*Hi*Wi*A, 4)
            anchor_deltas_pred_i = (
                anchor_deltas_pred_i.view(N, -1, 4, Hi, Wi).permute(0, 3, 4, 1, 2).reshape(-1, 4)
            )
            # Concatenate anchors to shape (N*Hi*Wi*A, 4)
            anchors_i = torch.cat([a.bbox for a in anchors_i], dim=0).reshape(-1, 4)
            proposals_i = self.box2box_transform.apply_deltas(anchor_deltas_pred_i, anchors_i)
            # Append feature map proposals with shape (N, Hi*Wi*A, 4)
            proposals.append(proposals_i.view(N, -1, 4))
        return proposals

    def predict_objectness_logits(self):
        """
        Return objectness logits in the same format as the proposals returned by
        :meth:`predict_proposals`.

        Returns:
            objectness_logits_pred (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A).
        """
        objectness_logits_pred = [
            # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).reshape(self.num_images, -1)
            for score in self.objectness_logits_pred
        ]
        return objectness_logits_pred
