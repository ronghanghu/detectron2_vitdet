import torch
import torch.nn.functional as F
from torch import nn

from torch_detectron.layers import nms as box_nms
from torch_detectron.structures.bounding_box import BoxList

from ..box_coder import BoxCoder
from .rpn import _clip_boxes_to_image


def box_results_with_nms_and_limit(
    scores, boxes, score_thresh=0.05, nms=0.5, detections_per_img=100
):
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).
    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.
    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = scores.shape[1]
    cls_boxes = []
    cls_scores = []
    labels = []
    device = scores.device
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = scores[:, j] > score_thresh
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
        keep = box_nms(boxes_j, scores_j, nms)
        cls_boxes.append(boxes_j[keep])
        cls_scores.append(scores_j[keep])
        # TODO see why we need the device argument
        labels.append(torch.full_like(keep, j, device=device))

    cls_scores = torch.cat(cls_scores, dim=0)
    cls_boxes = torch.cat(cls_boxes, dim=0)
    labels = torch.cat(labels, dim=0)
    number_of_detections = len(cls_scores)

    # Limit to max_per_image detections **over all classes**
    if number_of_detections > detections_per_img > 0:
        image_thresh, _ = torch.kthvalue(
            cls_scores.cpu(), number_of_detections - detections_per_img + 1
        )
        keep = cls_scores >= image_thresh.item()
        keep = torch.nonzero(keep)
        keep = keep.squeeze(1) if keep.numel() else keep
        cls_boxes = cls_boxes[keep]
        cls_scores = cls_scores[keep]
        labels = labels[keep]
    return cls_scores, cls_boxes, labels


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=BoxCoder(weights=(10., 10., 5., 5.)),
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        self.box_coder = box_coder

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size[::-1] for box in boxes]
        boxes_per_image = [box.bbox.size(0) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, proposal, (height, width) in zip(class_prob, proposals, image_shapes):
            clipped_proposal = _clip_boxes_to_image(proposal, height, width)
            cls_scores, cls_boxes, labels = box_results_with_nms_and_limit(
                prob,
                clipped_proposal,
                self.score_thresh,
                self.nms,
                self.detections_per_img,
            )
            bbox = BoxList(cls_boxes, (width, height), mode="xyxy")
            bbox.add_field("scores", cls_scores)
            bbox.add_field("labels", labels)
            bbox.add_field("level", torch.zeros_like(labels))
            results.append(bbox)

        return results
