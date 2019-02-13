import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import cat, smooth_l1_loss
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms, cat_boxlist


"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of classes including both background and foreground. E.g., if there
        are 80 foreground classes (as in COCO), then K = 81.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    class_logits_pred: predicted class scores in [-inf, +inf]; use
        softmax(class_logits_pred) to estimate P(class).

    classes_gt: ground-truth classification labels in [0, K), where 0 represents
        the background class and [1, K) represent foreground object classes.

    proposal_deltas_pred: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    proposal_deltas_gt: ground-truth box2box transform deltas
"""


def fast_rcnn_losses(classes_gt, proposal_deltas_gt, class_logits_pred, proposal_deltas_pred):
    """
    Compute the classification and box delta losses defined in the Fast R-CNN paper.

    Args:
        classes_gt (Tensor): A tensor of shape (R,) storing ground-truth classification
            labels in [0, K)
        proposal_deltas_gt (Tensor): shape (R, 4), row i represents ground-truth box2box
            transform targets (dx, dy, dw, dh) that map object instance i to its matched
            ground-truth box.
        class_logits_pred (Tensor): A tensor for shape (R, K) storing predicted classification
            logits for the K-way classification problem. Each row corresponds to a predicted
            object instance.
        proposal_deltas_pred (Tensor): shape (R, 4 * K), each row stores a list of class-specific
            predicted box2box transform [dx_0, dy_0, dw_0, dh_0, ..., dx_k, dy_k, dw_k, dh_k, ...]
            for each class k in [0, K). (Predictions for the background class, k = 0, are
            meaningless.)

    Returns:
        loss_cls, loss_box_reg (Tensor): Scalar loss values.
    """
    device = class_logits_pred.device

    loss_cls = F.cross_entropy(class_logits_pred, classes_gt, reduction="mean")

    # Box delta loss is only computed between the prediction for the gt class k
    # (if k > 0) and the target; there is no loss defined on predictions for
    # non-gt classes and background.
    # Empty fg_inds produces a valid loss of zero as long as the size_average
    # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
    # and would produce a nan loss).
    fg_inds = torch.nonzero(classes_gt > 0).squeeze(1)
    fg_classes_gt = classes_gt[fg_inds]
    # proposal_deltas_pred for class k are located in columns [4 * k : 4 * k + 4]
    gt_class_cols = 4 * fg_classes_gt[:, None] + torch.tensor([0, 1, 2, 3], device=device)

    loss_box_reg = smooth_l1_loss(
        proposal_deltas_pred[fg_inds[:, None], gt_class_cols],
        proposal_deltas_gt[fg_inds],
        size_average=False,
        beta=1,
    )
    # The loss is normalized using the total number of regions (R), not the number
    # of foreground regions even though the box regression loss is only defined on
    # foreground regions. Why? Because doing so gives equal training influence to
    # each foreground example. To see how, consider two different minibatches:
    #  (1) Contains a single foreground region
    #  (2) Contains 100 foreground regions
    # If we normalize by the number of foreground regions, the single example in
    # minibatch (1) will be given 100 times as much influence as each foreground
    # example in minibatch (2). Normalizing by the total number of regions, R,
    # means that the single example in minibatch (1) and each of the 100 examples
    # in minibatch (2) are given equal influence.
    loss_box_reg = loss_box_reg / classes_gt.numel()

    return loss_cls, loss_box_reg


def fast_rcnn_inference(boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific boxes for each image.
            Element i has shape (Ri, K * 4), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputs.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputs.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        list[BoxList]: A list of N box lists, one for each image in the batch, that stores the
            topk most confidence detections.
    """
    return [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]


def fast_rcnn_inference_single_image(
    boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    num_classes = scores.shape[1]
    # convert to boxlist to use the `clip` function ...
    boxlist = BoxList(boxes.reshape(-1, 4), image_shape, mode="xyxy")
    boxlist = boxlist.clip_to_image(remove_empty=False)
    boxes = boxlist.bbox.view(-1, num_classes, 4)

    device = scores.device
    result = []
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    inds_all = scores > score_thresh
    for j in range(1, num_classes):
        inds = inds_all[:, j].nonzero().squeeze(1)
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j, :]
        # image_size is not used, fill -1
        boxlist_for_class = BoxList(boxes_j, image_shape, mode="xyxy")
        boxlist_for_class.add_field("scores", scores_j)
        boxlist_for_class = boxlist_nms(boxlist_for_class, nms_thresh, score_field="scores")
        num_labels = len(boxlist_for_class)
        boxlist_for_class.add_field(
            "classes_pred", torch.full((num_labels,), j, dtype=torch.int64, device=device)
        )
        result.append(boxlist_for_class)

    result = cat_boxlist(result)
    number_of_detections = len(result)

    # Limit to max_per_image detections **over all classes**
    if number_of_detections > topk_per_image > 0:
        cls_scores = result.get_field("scores")
        image_thresh, _ = torch.kthvalue(
            cls_scores.cpu(), number_of_detections - topk_per_image + 1
        )
        keep = cls_scores >= image_thresh.item()
        keep = torch.nonzero(keep).squeeze(1)
        result = result[keep]
    return result


class FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """

    def __init__(
        self, box2box_transform, class_logits_pred, proposal_deltas_pred, proposal_box_lists
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform): :class:`Box2BoxTransform` instance for
                proposal-to-detection tranformations.
            class_logits_pred (Tensor): A tensor of shape (R, K) storing the predicted class
                logits for all R predicted object instances.
            proposal_deltas_pred (Tensor): A tensor of shape (R, K * 4) storing the predicted
                deltas that transform proposals into final box detections.
            proposal_box_lists (list[BoxList]): A list of N BoxLists, where BoxList i stores the
                proposals for image i. When training, each BoxList has ground-truth labels
                stored in the field "classes_gt" and "boxes_gt".
        """
        self.box2box_transform = box2box_transform
        self.num_images = len(proposal_box_lists)
        self.image_shapes = [
            box_list.size for box_list in proposal_box_lists
        ]  # NB: (width, height)
        self.num_preds_per_image = [len(box_list) for box_list in proposal_box_lists]
        self.num_classes = class_logits_pred.shape[1]
        self.class_logits_pred = class_logits_pred
        self.proposal_deltas_pred = proposal_deltas_pred

        # cat(..., dim=0) concatenates over all images in the batch
        self.proposals = cat([box_list.bbox for box_list in proposal_box_lists], dim=0)

        # The following fields should exist only when training.
        if proposal_box_lists[0].has_field("boxes_gt"):
            self.boxes_gt = cat(
                [box_list.get_field("boxes_gt") for box_list in proposal_box_lists], dim=0
            )
        if proposal_box_lists[0].has_field("classes_gt"):
            self.classes_gt = cat(
                [box.get_field("classes_gt") for box in proposal_box_lists], dim=0
            )

    def losses(self):
        """
        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        proposal_deltas_gt = self.box2box_transform.get_deltas(self.proposals, self.boxes_gt)
        loss_cls, loss_box_reg = fast_rcnn_losses(
            self.classes_gt, proposal_deltas_gt, self.class_logits_pred, self.proposal_deltas_pred
        )
        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific boxes for each image.
                Element i has shape (Ri, K * 4), where Ri is the number of predicted objects
                for image i.
        """
        boxes = self.box2box_transform.apply_deltas(self.proposal_deltas_pred, self.proposals)
        return boxes.split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K), where Ri is the number of predicted objects
                for image i.
        """
        probs = F.softmax(self.class_logits_pred, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)


# TODO a better name?
class FastRCNNOutputHead(nn.Module):
    """
    2 FC layers that does bbox regression and classification, respectively.
    """

    def __init__(self, input_size, num_classes):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int):
        """
        super(FastRCNNOutputHead, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        self.cls_score = nn.Linear(input_size, num_classes)
        self.bbox_pred = nn.Linear(input_size, num_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas
