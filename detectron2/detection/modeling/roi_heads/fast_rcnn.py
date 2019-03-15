import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import cat, nms, smooth_l1_loss
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage


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

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K), where 0 represents
        the background class and [1, K) represent foreground object classes.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    proposal_deltas_gt: ground-truth box2box transform deltas
"""


def fast_rcnn_losses(gt_classes, proposal_deltas_gt, pred_class_logits, pred_proposal_deltas):
    """
    Compute the classification and box delta losses defined in the Fast R-CNN paper.

    Args:
        gt_classes (Tensor): A tensor of shape (R,) storing ground-truth classification
            labels in [0, K)
        proposal_deltas_gt (Tensor): shape (R, 4), row i represents ground-truth box2box
            transform targets (dx, dy, dw, dh) that map object instance i to its matched
            ground-truth box.
        pred_class_logits (Tensor): A tensor for shape (R, K) storing predicted classification
            logits for the K-way classification problem. Each row corresponds to a predicted
            object instance.
        pred_proposal_deltas (Tensor): shape (R, 4 * K), each row stores a list of class-specific
            predicted box2box transform [dx_0, dy_0, dw_0, dh_0, ..., dx_k, dy_k, dw_k, dh_k, ...]
            for each class k in [0, K). (Predictions for the background class, k = 0, are
            meaningless.)

    Returns:
        loss_cls, loss_box_reg (Tensor): Scalar loss values.
    """
    device = pred_class_logits.device

    loss_cls = F.cross_entropy(pred_class_logits, gt_classes, reduction="mean")

    # Box delta loss is only computed between the prediction for the gt class k
    # (if k > 0) and the target; there is no loss defined on predictions for
    # non-gt classes and background.
    # Empty fg_inds produces a valid loss of zero as long as the size_average
    # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
    # and would produce a nan loss).
    fg_inds = torch.nonzero(gt_classes > 0).squeeze(1)
    fg_gt_classes = gt_classes[fg_inds]
    # pred_proposal_deltas for class k are located in columns [4 * k : 4 * k + 4]
    gt_class_cols = 4 * fg_gt_classes[:, None] + torch.tensor([0, 1, 2, 3], device=device)

    loss_box_reg = smooth_l1_loss(
        pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
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
    loss_box_reg = loss_box_reg / gt_classes.numel()

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
        list[Boxes]: A list of N box lists, one for each image in the batch, that stores the
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
    # convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_classes, 4)

    device = scores.device
    results = []
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    inds_all = scores > score_thresh
    for j in range(1, num_classes):
        inds = inds_all[:, j].nonzero().squeeze(1)
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j, :]
        # image_size is not used, fill -1
        keep = nms(boxes_j, scores_j, nms_thresh)
        boxes_j = boxes_j[keep]
        num_labels = len(keep)
        classes_j = torch.full((num_labels,), j, dtype=torch.int64, device=device)

        result_j = Instances(image_shape)
        result_j.pred_boxes = Boxes(boxes_j)
        result_j.scores = scores_j[keep]
        result_j.pred_classes = classes_j
        results.append(result_j)

    results = Instances.cat(results)
    number_of_detections = len(results)

    # Limit to max_per_image detections **over all classes**
    if number_of_detections > topk_per_image > 0:
        cls_scores = results.scores
        image_thresh, _ = torch.kthvalue(
            cls_scores.cpu(), number_of_detections - topk_per_image + 1
        )
        keep = cls_scores >= image_thresh.item()
        keep = torch.nonzero(keep).squeeze(1)
        results = results[keep]
    return results


class FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """

    def __init__(self, box2box_transform, pred_class_logits, pred_proposal_deltas, proposals):
        """
        Args:
            box2box_transform (Box2BoxTransform): :class:`Box2BoxTransform` instance for
                proposal-to-detection tranformations.
            pred_class_logits (Tensor): A tensor of shape (R, K) storing the predicted class
                logits for all R predicted object instances.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * 4) storing the predicted
                deltas that transform proposals into final box detections.
            proposals (list[Instances]): A list of N Instancess, where Instances i stores the
                proposals for image i. When training, each Instances has ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.num_classes = pred_class_logits.shape[1]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas

        # cat(..., dim=0) concatenates over all images in the batch
        self.proposals = Boxes.cat([p.proposal_boxes for p in proposals])

        # The following fields should exist only when training.
        if proposals[0].has("gt_boxes"):
            self.gt_boxes = Boxes.cat([p.gt_boxes for p in proposals])
            assert proposals[0].has("gt_classes")
            self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)

        fg_inds = self.gt_classes > 0
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == 0).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
        storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
        storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)

    def losses(self):
        """
        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        self._log_accuracy()
        proposal_deltas_gt = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        loss_cls, loss_box_reg = fast_rcnn_losses(
            self.gt_classes, proposal_deltas_gt, self.pred_class_logits, self.pred_proposal_deltas
        )
        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific boxes for each image.
                Element i has shape (Ri, K * 4), where Ri is the number of predicted objects
                for image i.
        """
        boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas, self.proposals.tensor
        )
        return boxes.split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K), where Ri is the number of predicted objects
                for image i.
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
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
