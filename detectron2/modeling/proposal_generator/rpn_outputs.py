import numpy as np
import torch
import torch.nn.functional as F
from borc.nn import smooth_l1_loss

from detectron2.layers import cat, nms
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

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

    pred_objectness_logits: predicted objectness scores in [-inf, +inf]; use
        sigmoid(pred_objectness_logits) to estimate P(object).

    gt_objectness_logits: ground-truth binary classification labels for objectness

    pred_anchor_deltas: predicted box2box transform deltas

    gt_anchor_deltas: ground-truth box2box transform deltas
"""


def find_top_rpn_proposals(
    proposals,
    pred_objectness_logits,
    images,
    nms_thresh,
    pre_nms_topk,
    post_nms_topk,
    min_box_side_len,
    training,
):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps if `training` is True,
    otherwise, returns the highest `post_nms_topk` scoring proposals for each
    feature map.

    Args:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
            All proposal predictions on the feature maps.
        pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
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
        proposals (list[Instances]): list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i.
    """
    image_sizes = images.image_sizes  # in (h, w) order

    proposals = [
        _find_top_rpn_proposals_single_feature_map(
            proposals_i,
            pred_objectness_logits_i,
            image_sizes,
            nms_thresh,
            pre_nms_topk,
            post_nms_topk,
            min_box_side_len,
        )
        for proposals_i, pred_objectness_logits_i in zip(proposals, pred_objectness_logits)
    ]  # (L, N) Instances.

    if len(proposals) == 1:
        # Single feature map, no need to merge
        return proposals[0]

    # Merge proposals from all levels within each image
    proposals = list(zip(*proposals))  # transpose to (N, L) Instances
    # concat proposals across feature maps
    proposals = [Instances.cat(instances) for instances in proposals]  # cat to (N, ) Instances
    num_images = len(proposals)

    # In Detectron1, there was different behavior during training vs. testing.
    # (https://github.com/facebookresearch/Detectron/issues/459)
    # During training, topk is over the proposals from *all* images in the training batch.
    # During testing, it is over the proposals for each image separately.
    # As a result, the training behavior becomes batch-dependent,
    # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
    # This bug is addressed in Detectron2 to make the behavior independent of batch size.
    for i in range(num_images):
        pred_objectness_logits = proposals[i].objectness_logits
        topk = min(post_nms_topk, len(pred_objectness_logits))
        _, inds_sorted = torch.topk(pred_objectness_logits, topk, dim=0, sorted=True)
        proposals[i] = proposals[i][inds_sorted]
    return proposals


def _find_top_rpn_proposals_single_feature_map(
    proposals,
    pred_objectness_logits,
    image_sizes,
    nms_thresh,
    pre_nms_topk,
    post_nms_topk,
    min_box_side_len,
):
    """
    Select the `pre_nms_topk` highest scoring proposals, applies NMS, clip
    proposals, remove small boxes, and finally return the `post_nms_topk`
    highest scoring proposals from a single feature map.

    Args:
        image_sizes: N (height, width) tuples.
        Otherwise, see `sample_rpn_proposals`.

    Returns:
        list[Instancess]: list of N Instances. Instances i stores post_nms_topk object
            proposals for image i, with field "proposal_boxes" and "objectness_logits".
    """
    N, Hi_Wi_A = pred_objectness_logits.shape
    device = pred_objectness_logits.device
    assert proposals.shape[:2] == (N, Hi_Wi_A)

    pre_nms_topk = min(pre_nms_topk, Hi_Wi_A)
    pred_objectness_logits, topk_idx = pred_objectness_logits.topk(pre_nms_topk, dim=1)

    batch_idx = torch.arange(N, device=device)[:, None]
    proposals = proposals[batch_idx, topk_idx]  # shape is now (N, pre_nms_topk, 4)

    result = []
    for proposals_i, pred_objectness_logits_i, image_size_i in zip(
        proposals, pred_objectness_logits, image_sizes
    ):
        """
        proposals_i: tensor of shape (topk, 4)
        pred_objectness_logits_i: top-k pred_objectness_logits_i of shape (topk, )
        image_size_i: image (height, width)
        """
        boxes = Boxes(proposals_i)

        boxes.clip(image_size_i)
        keep = boxes.nonempty(
            threshold=min_box_side_len
        )  # is min_box_side_len still an interesting config or can we just use 0?
        boxes = boxes[keep]
        scores = pred_objectness_logits_i[keep]
        keep = nms(boxes.tensor, scores, nms_thresh)
        if post_nms_topk > 0:
            # outputs of nms() are already sorted by scores
            keep = keep[:post_nms_topk]

        instances = Instances(image_size_i)
        instances.proposal_boxes = boxes[keep]
        instances.objectness_logits = scores[keep]
        result.append(instances)
    return result


def rpn_losses(
    gt_objectness_logits,
    gt_anchor_deltas,
    pred_objectness_logits,
    pred_anchor_deltas,
    smooth_l1_beta,
):
    """
    Args:
        gt_objectness_logits (Tensor): shape (N,), each element in {-1, 0, 1} representing
            ground-truth objectness labels with: -1 = ignore; 0 = not object; 1 = object.
        gt_anchor_deltas (Tensor): shape (N, 4), row i represents ground-truth
            box2box transform targets (dx, dy, dw, dh) that map anchor i to its
            matched ground-truth box.
        pred_objectness_logits (Tensor): shape (N,), each element is a predicted objectness
            logit.
        pred_anchor_deltas (Tensor): shape (N, 4), each row is a predicted box2box
            transform (dx, dy, dw, dh).
        smooth_l1_beta (float): The transition point between L1 and L2 loss in
            the smooth L1 loss function. When set to 0, the loss becomes L1. When
            set to +inf, the loss becomes constant 0.

    Returns:
        objectness_loss, localization_loss, both unnormalized (summed over samples).
    """
    pos_masks = gt_objectness_logits == 1
    localization_loss = smooth_l1_loss(
        pred_anchor_deltas[pos_masks], gt_anchor_deltas[pos_masks], smooth_l1_beta, reduction="sum"
    )

    valid_masks = gt_objectness_logits >= 0
    objectness_loss = F.binary_cross_entropy_with_logits(
        pred_objectness_logits[valid_masks],
        gt_objectness_logits[valid_masks].to(torch.float32),
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
        pred_objectness_logits,
        pred_anchor_deltas,
        anchors,
        boundary_threshold=0,
        gt_boxes=None,
        smooth_l1_beta=0.0,
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
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for anchors.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, A*4, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
            anchors (list[list[Boxes]]): A list of N elements. Each element is a list of L
                Boxes. The Boxes at (n, l) stores the entire anchor array for feature map l in image
                n (i.e. the cell anchors repeated over all locations in feature map (n, l)).
            boundary_threshold (int): if >= 0, then anchors that extend beyond the image
                boundary by more than boundary_thresh are not used in training. Set to a very large
                number or < 0 to disable this behavior. Only needed in training.
            gt_boxes (list[Boxes], optional): A list of N elements. Element i a Boxes storing
                the ground-truth ("gt") boxes for image i.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.pred_objectness_logits = pred_objectness_logits
        self.pred_anchor_deltas = pred_anchor_deltas
        self.anchors = anchors
        self.gt_boxes = gt_boxes
        self.num_feature_maps = len(pred_objectness_logits)
        self.num_images = len(images)
        self.image_sizes = images.image_sizes
        self.boundary_threshold = boundary_threshold
        self.smooth_l1_beta = smooth_l1_beta

    def _get_ground_truth(self):
        """
        Returns:
            gt_objectness_logits: list of N tensors. Tensor i is a vector whose length is the
                total number of anchors in image i (i.e., len(anchors[i])). Label values are
                in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
            gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), 4).
        """
        gt_objectness_logits = []
        gt_anchor_deltas = []
        # Concatenate anchors from all feature maps into a single Boxes per image
        anchors = [Boxes.cat(anchors_i) for anchors_i in self.anchors]
        for image_size_i, anchors_i, gt_boxes_i in zip(self.image_sizes, anchors, self.gt_boxes):
            """
            image_size_i: (h, w) for the i-th image
            anchors_i: anchors for i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """
            match_quality_matrix = pairwise_iou(gt_boxes_i, anchors_i)
            matched_idxs, gt_objectness_logits_i = self.anchor_matcher(match_quality_matrix)

            if self.boundary_threshold >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors_i.inside_box(image_size_i, self.boundary_threshold)
                gt_objectness_logits_i[~anchors_inside_image] = -1

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                gt_anchor_deltas_i = torch.zeros_like(anchors_i.tensor)
            else:
                # TODO wasted computation for ignored boxes
                matched_gt_boxes = gt_boxes_i[matched_idxs]
                gt_anchor_deltas_i = self.box2box_transform.get_deltas(
                    anchors_i.tensor, matched_gt_boxes.tensor
                )

            gt_objectness_logits.append(gt_objectness_logits_i)
            gt_anchor_deltas.append(gt_anchor_deltas_i)

        return gt_objectness_logits, gt_anchor_deltas

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
                label, self.batch_size_per_image, self.positive_fraction, 0
            )
            # Fill with the ignore label (-1), then set positive and negative labels
            label.fill_(-1)
            label.scatter_(0, pos_idx, 1)
            label.scatter_(0, neg_idx, 0)
            return label

        gt_objectness_logits, gt_anchor_deltas = self._get_ground_truth()
        """
        gt_objectness_logits: list of N tensors. Tensor i is a vector whose length is the
            total number of anchors in image i (i.e., len(anchors[i]))
        gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), 4).
        """

        # Collect all objectness labels and delta targets over feature maps and images
        # The final ordering is L, N, H, W, A from slowest to fastest axis.
        num_anchors_per_map = [np.prod(x.shape[1:]) for x in self.pred_objectness_logits]
        num_anchors_per_image = sum(num_anchors_per_map)

        # Stack to: (N, num_anchors_per_image)
        gt_objectness_logits = torch.stack(
            [resample(label) for label in gt_objectness_logits], dim=0
        )

        # Log the number of positive/negative anchors per-image that's used in training
        num_pos_anchors = (gt_objectness_logits == 1).nonzero().size(0)
        num_neg_anchors = (gt_objectness_logits == 0).nonzero().size(0)
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / self.num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / self.num_images)

        assert gt_objectness_logits.shape[1] == num_anchors_per_image
        # Split to tuple of L tensors, each with shape (N, num_anchors_per_map)
        gt_objectness_logits = torch.split(gt_objectness_logits, num_anchors_per_map, dim=1)
        # Concat from all feature maps
        gt_objectness_logits = cat([x.flatten() for x in gt_objectness_logits], dim=0)

        # Stack to: (N, num_anchors_per_image, 4)
        gt_anchor_deltas = torch.stack(gt_anchor_deltas, dim=0)
        assert gt_anchor_deltas.shape[1] == num_anchors_per_image
        # Split to tuple of L tensors, each with shape (N, num_anchors_per_image)
        gt_anchor_deltas = torch.split(gt_anchor_deltas, num_anchors_per_map, dim=1)
        # Concat from all feature maps
        gt_anchor_deltas = cat([x.reshape(-1, 4) for x in gt_anchor_deltas], dim=0)

        # Collect all objectness logits and delta predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W, A from slowest to fastest axis.
        pred_objectness_logits = cat(
            [
                # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N*Hi*Wi*A, )
                x.permute(0, 2, 3, 1).flatten()
                for x in self.pred_objectness_logits
            ],
            dim=0,
        )
        pred_anchor_deltas = cat(
            [
                # Reshape: (N, A*4, Hi, Wi) -> (N, A, 4, Hi, Wi) -> (N, Hi, Wi, A, 4)
                #          -> (N*Hi*Wi*A, 4)
                x.view(x.shape[0], -1, 4, x.shape[-2], x.shape[-1])
                .permute(0, 3, 4, 1, 2)
                .reshape(-1, 4)
                for x in self.pred_anchor_deltas
            ],
            dim=0,
        )

        objectness_loss, localization_loss = rpn_losses(
            gt_objectness_logits,
            gt_anchor_deltas,
            pred_objectness_logits,
            pred_anchor_deltas,
            self.smooth_l1_beta,
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
        for anchors_i, pred_anchor_deltas_i in zip(anchors, self.pred_anchor_deltas):
            N, _, Hi, Wi = pred_anchor_deltas_i.shape
            # Reshape: (N, A*4, Hi, Wi) -> (N, A, 4, Hi, Wi) -> (N, Hi, Wi, A, 4) -> (N*Hi*Wi*A, 4)
            pred_anchor_deltas_i = (
                pred_anchor_deltas_i.view(N, -1, 4, Hi, Wi).permute(0, 3, 4, 1, 2).reshape(-1, 4)
            )
            # Concatenate all anchors to shape (N*Hi*Wi*A, 4)
            anchors_i = Boxes.cat(anchors_i)
            proposals_i = self.box2box_transform.apply_deltas(
                pred_anchor_deltas_i, anchors_i.tensor
            )
            # Append feature map proposals with shape (N, Hi*Wi*A, 4)
            proposals.append(proposals_i.view(N, -1, 4))
        return proposals

    def predict_objectness_logits(self):
        """
        Return objectness logits in the same format as the proposals returned by
        :meth:`predict_proposals`.

        Returns:
            pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A).
        """
        pred_objectness_logits = [
            # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).reshape(self.num_images, -1)
            for score in self.pred_objectness_logits
        ]
        return pred_objectness_logits
