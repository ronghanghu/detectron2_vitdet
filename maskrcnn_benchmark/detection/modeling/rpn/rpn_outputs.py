import numpy as np
import torch
import torch.nn.functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import (
    boxlist_iou,
    boxlist_nms,
    cat_boxlist,
    remove_small_boxes,
)

from ..balanced_positive_negative_sampler import sample_with_positive_fraction
from ..matcher import Matcher


# TODO: comments for future refactoring of this module
#
# From @yuxinwu:
# On a high level I think the RPNOutputs class encapsulates too many functionalities.
# This makes it hard to, for example, reuse the proposal sampling & add_gt_proposals
# logic for models with pre-computed proposals.
# The analogous class to FastRCNNoutputs is to decode the direct outputs of a RPN head
# network. Post processing afterwards may be moved outside for reuse in models with
# pre-computed proposals.
#
# From @rbg:
# This code involves a significant amount of tensor reshaping and permuting. Look for
# ways to simplify this.


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


class RPNOutputs(object):
    """
    Shape shorthand in this class:
        N: number of images in the minibatch
        L: number of feature maps per image
        A: number of cell anchors (must be the same for all feature maps)
        Hi, Wi: height and width of the i-th feature map
    """

    def __init__(
        self,
        box2box_transform,
        anchor_matcher,
        batch_size_per_image,
        positive_fraction,
        images,
        objectness,
        anchor_deltas,
        anchors,
        targets=None,
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
            objectness (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, A, Hi, Wi) representing the predicted objectness logits for anchors.
            anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, A*4, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
            anchors (list[list[BoxLists]]): A list of N elements. Each element is a list of L
                BoxLists. BoxList (n, l) stores the entire anchor array for feature map l in image
                n (i.e. the cell anchors repeated over all locations in feature map (n, l)).
            targets (list[BoxList, optional): A list of N elements. Element i a BoxList storing
                the ground-truth boxes for image i.
        """
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.images = images
        self.objectness = objectness
        self.anchor_deltas = anchor_deltas
        self.anchors = anchors
        self.targets = targets
        self.num_feature_maps = len(objectness)
        self.num_images = len(images)

    def prepare_targets(self):
        labels = []
        regression_targets = []
        # Concatenate anchors from all feature maps into a single BoxList per image
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in self.anchors]
        for anchors_per_image, targets_per_image in zip(anchors, self.targets):
            match_quality_matrix = boxlist_iou(targets_per_image, anchors_per_image)
            matched_idxs = self.anchor_matcher(match_quality_matrix)
            # NB: need to clamp the indices because matched_idxs can be <0
            matched_targets = targets_per_image.bbox[matched_idxs.clamp(min=0)]

            labels_per_image = (matched_idxs >= 0).to(dtype=torch.int32)
            # discard anchors that go out of the boundaries of the image
            labels_per_image[~anchors_per_image.get_field("visibility")] = -1
            # discard indices that are neither foreground or background
            labels_per_image[matched_idxs == Matcher.BETWEEN_THRESHOLDS] = -1

            # compute regression targets
            # TODO wasted computation for ignored boxes
            regression_targets_per_image = self.box2box_transform.get_deltas(
                anchors_per_image.bbox, matched_targets
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def losses(self):
        labels, regression_targets = self.prepare_targets()
        """
        labels: list of N tensors. Tensor i is a vector whose length is the total number of
            anchors in image i (i.e., len(anchors[i])). Label values are in {-1, 0, 1},
            with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
        regression_targets: list of N tensors. Tensor i has shape (len(anchors[i]), 4).
        """

        def resample(label):
            """
            Randomly sample a subset of positive and negative examples by overwritting
            the label vector to the ignore value (-1) for all elements that are not
            included in the sample.
            """
            pos_idx, neg_idx = sample_with_positive_fraction(
                label, self.batch_size_per_image, self.positive_fraction
            )
            # Fill with the ignore label (-1), then set positive and negative labels
            label.fill_(-1)
            label.scatter_(0, pos_idx, 1)
            label.scatter_(0, neg_idx, 0)
            return label

        labels = torch.stack([resample(label) for label in labels], dim=0)  # N x #all_anchors
        num_anchors_per_map = [np.prod(x.shape[1:]) for x in self.objectness]
        total_num_anchors = sum(num_anchors_per_map)
        assert labels.shape[1] == total_num_anchors
        labels = torch.split(labels, num_anchors_per_map, dim=1)  # N x (HxWx#anchors_per_map)

        regression_targets = torch.stack(regression_targets, dim=0)
        assert regression_targets.shape[1] == total_num_anchors
        regression_targets = torch.split(regression_targets, num_anchors_per_map, dim=1)

        # For each feature map, permute the outputs to a format that is compatible with the labels.
        obj_losses, box_losses = [], []
        for map_idx in range(self.num_feature_maps):
            labels_per_map, objectness_per_map, anchor_deltas_per_map, regression_targets_per_map = (
                labels[map_idx],
                self.objectness[map_idx],
                self.anchor_deltas[map_idx],
                regression_targets[map_idx],
            )
            N, A, H, W = objectness_per_map.shape
            objectness_per_map = objectness_per_map.permute(0, 2, 3, 1).flatten()
            anchor_deltas_per_map = anchor_deltas_per_map.view(N, -1, 4, H, W)
            anchor_deltas_per_map = anchor_deltas_per_map.permute(0, 3, 4, 1, 2).reshape(-1, 4)

            obj_loss, box_loss = rpn_losses(
                labels_per_map.flatten(),
                regression_targets_per_map.reshape(-1, 4),
                objectness_per_map,
                anchor_deltas_per_map,
            )
            obj_losses.append(obj_loss)
            box_losses.append(box_loss)

        obj_loss, box_loss = sum(obj_losses), sum(box_losses)
        normalizer = self.batch_size_per_image * self.num_images
        loss_objectness = obj_loss / normalizer
        loss_box_reg = box_loss / normalizer
        losses = {"loss_rpn_cls": loss_objectness, "loss_rpn_box": loss_box_reg}
        return losses

    def add_gt_proposals(self, proposals):
        """
        Augment `proposals` with ground-truth boxes.

        Args:
            proposals (list[BoxList]): list of N elements. Element i is a BoxList
                representing the proposals for image i, as returned by :meth:`proposals`.

        Returns:
            list[BoxList]: list of N BoxLists.
        """
        if self.targets is None:
            return proposals

        device = proposals[0].bbox.device
        gt_boxes = [target.copy_with_fields([]) for target in self.targets]
        # Later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.add_field("objectness", torch.ones(len(gt_box), device=device))

        proposals = [
            cat_boxlist((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def _transform_anchors_to_proposals(self):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.
        Objectness scores are returned in a format that allows for easier indexing.

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4)
            objectness (list[Tensor]): list of L tensors. Tensor i has shape (N, Hi*Wi*A)
        """
        proposals = []
        # Transpose anchors from images-by-feature-maps (N, L) to feature-maps-by-images (L, N)
        anchors = list(zip(*self.anchors))
        # For each feature map
        for anchors_i, deltas_i in zip(anchors, self.anchor_deltas):
            N, _, Hi, Wi = deltas_i.shape
            # Reshape: (N, A*4, Hi, Wi) -> (N, A, 4, Hi, Wi) -> (N, Hi, Wi, A, 4) -> (N*Hi*Wi*A, 4)
            deltas_i = deltas_i.view(N, -1, 4, Hi, Wi).permute(0, 3, 4, 1, 2).reshape(-1, 4)
            # Concatenate anchors to shape (N*Hi*Wi*A, 4)
            anchors_i = torch.cat([a.bbox for a in anchors_i], dim=0).reshape(-1, 4)
            proposals_i = self.box2box_transform.apply_deltas(deltas_i, anchors_i)
            # Append feature map proposals with shape (N, Hi*Wi*A, 4)
            proposals.append(proposals_i.view(N, -1, 4))

        objectness = [
            # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).reshape(self.num_images, -1)
            for score in self.objectness
        ]
        return proposals, objectness

    def proposals(self, nms_thresh, pre_nms_topk, post_nms_topk, min_size, training):
        """
        Return scored object proposals after applying box regression to the anchors, NMS,
        taking the top k, etc.

        Args:
            nms_thresh (float): IoU threshold to use for NMS
            pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
                When RPN is run on multiple feature maps (as in FPN) this number is per
                feature map.
            post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
                When RPN is run on multiple feature maps (as in FPN) this number is total,
                over all feature maps.
            min_size (float): minimum proposal size in pixels (absolute units wrt input images).
            training (bool): True if proposals are to be used in training, otherwise False.

        Returns:
            proposals (list[BoxLists]): list of N BoxLists. BoxList i stores post_nms_topk object
                proposals for image i.
        """
        proposals, objectness = self._transform_anchors_to_proposals()
        proposals = self._post_process_proposals(
            proposals,
            objectness,
            self.images.image_sizes,
            nms_thresh=nms_thresh,
            pre_nms_topk=pre_nms_topk,
            post_nms_topk=post_nms_topk,
            min_size=min_size,
            training=training,
        )
        return proposals

    def _post_process_proposals(
        self,
        boxes,
        objectness,
        image_sizes,
        nms_thresh,
        pre_nms_topk,
        post_nms_topk,
        min_size=0,
        training=False,
    ):
        # Convert from (h, w) to (w, h)
        # FIXME: make size order consistent everywhere
        image_sizes = [size[::-1] for size in image_sizes]
        if self.num_feature_maps > 1:
            return self._post_process_proposals_multi_feature_maps(
                boxes,
                objectness,
                image_sizes,
                nms_thresh,
                pre_nms_topk,
                post_nms_topk,
                min_size,
                training,
            )
        else:
            return self._post_process_proposals_single_feature_map(
                boxes[0],
                objectness[0],
                image_sizes,
                nms_thresh,
                pre_nms_topk,
                post_nms_topk,
                min_size,
            )

    def _post_process_proposals_multi_feature_maps(
        self,
        multi_map_boxes,
        multi_map_objectness,
        image_sizes,
        nms_thresh,
        pre_nms_topk,
        post_nms_topk,
        min_size=0,
        training=False,
    ):
        """
        Generate RPN proposals from multiple feature maps (e.g., as in FPN).
        Currently it does so by merging the outputs of `post_process_proposals`.

        Args:
            multi_map_boxes(list[Tensor]): list of L tensors. Each tensor has size (N, Hi*Wi*A, 4).
            multi_map_objectness(list[Tensor]): list of L tensors. Each tensor has size
                (N, Hi*Wi*A).
            image_sizes: N (w, h) tuples.

        Returns:
            list[BoxList], with N BoxList. Each BoxList is the proposals on the corresponding image.
        """
        # TODO an alternative is to merge them first, then do topk and NMS
        multi_map_proposals = [
            self._post_process_proposals_single_feature_map(
                boxes, objectness, image_sizes, nms_thresh, pre_nms_topk, post_nms_topk, min_size
            )
            for boxes, objectness in zip(multi_map_boxes, multi_map_objectness)
        ]  # (L, N) BoxList.
        multi_map_proposals = list(zip(*multi_map_proposals))  # (N, L) BoxList
        # concat proposals across feature maps
        proposals_per_img = [
            cat_boxlist(boxlist) for boxlist in multi_map_proposals
        ]  # (N, ) BoxList
        num_img = len(proposals_per_img)

        # Legacy bug to address -- there's different behavior during training vs. testing.
        # During training, topk is over *all* the proposals combined over all images.
        # During testing, it is over the proposals for each image separately.
        # TODO: resolve this difference and make it consistent. It should be per image,
        # and not per batch, see: https://github.com/facebookresearch/Detectron/issues/459.
        if training:
            objectness = torch.cat(
                [boxlist.get_field("objectness") for boxlist in proposals_per_img], dim=0
            )
            box_sizes = [len(boxlist) for boxlist in proposals_per_img]
            topk = min(post_nms_topk, len(objectness))
            _, inds_sorted = torch.topk(objectness, topk, dim=0, sorted=True)
            inds_mask = torch.zeros_like(objectness, dtype=torch.uint8)
            inds_mask[inds_sorted] = 1
            inds_mask = inds_mask.split(box_sizes)
            for i in range(num_img):
                proposals_per_img[i] = proposals_per_img[i][inds_mask[i]]
        else:
            for i in range(num_img):
                objectness = proposals_per_img[i].get_field("objectness")
                topk = min(post_nms_topk, len(objectness))
                _, inds_sorted = torch.topk(objectness, topk, dim=0, sorted=True)
                proposals_per_img[i] = proposals_per_img[i][inds_sorted]
        return proposals_per_img

    def _post_process_proposals_single_feature_map(
        self, boxes, objectness, image_sizes, nms_thresh, pre_nms_topk, post_nms_topk, min_size=0
    ):
        """
        Generate RPN proposals from RPN boxes at all locations.
        It does so by NMS and keeping the topk.

        Args:
            boxes: tensor of size (N, Hi*Wi*A, 4), decoded boxes, where N is the total number of anchors
                on the feature map.
            objectness: tensor of size (N, Hi*Wi*A), before sigmoid
            image_sizes: N (w, h) tuples.
            pre_nms_topk, post_nms_topk: int
            min_size: int

        Returns:
            list[BoxList], with B BoxList. Each BoxList is the proposals on the corresponding image.
        """
        N, Hi_Wi_A = objectness.shape
        device = objectness.device
        assert boxes.shape[:2] == (N, Hi_Wi_A)

        pre_nms_topk = min(pre_nms_topk, Hi_Wi_A)
        objectness, topk_idx = objectness.topk(pre_nms_topk, dim=1)

        batch_idx = torch.arange(N, device=device)[:, None]
        boxes = boxes[batch_idx, topk_idx]  # N, topk, 4

        result = []
        for proposals, scores, im_shape in zip(boxes, objectness, image_sizes):
            """
            proposals: tensor of shape (topk, 4)
            scores: top-k scores of shape (topk, )
            """

            boxlist = BoxList(proposals, im_shape, mode="xyxy")
            # TODO why using "field" at all? Avoid storing arbitrary states
            boxlist.add_field("objectness", scores)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, min_size)
            boxlist = boxlist_nms(boxlist, nms_thresh, topk=post_nms_topk, score_field="objectness")
            result.append(boxlist)
        return result
