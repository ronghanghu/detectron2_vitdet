import torch
import torch.nn.functional as F
from torch import nn

from torch_detectron.layers import nms as box_nms
from torch_detectron.structures.bounding_box import BoxList
from torch_detectron.structures.boxlist_ops import boxlist_nms

from ..box_coder import BoxCoder
from ..utils import cat_bbox


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
    inds_all = scores > score_thresh
    for j in range(1, num_classes):
        inds = inds_all[:, j].nonzero().squeeze(1)
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
        keep = torch.nonzero(keep).squeeze(1)
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
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )

        num_classes = class_prob.shape[1]
        label_ids = torch.arange(num_classes, device=class_prob.device)

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_img, image_shape in zip(class_prob, proposals, image_shapes):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape, label_ids)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = self.filter_results(boxlist, num_classes)
            results.append(boxlist)
        return results


    def prepare_boxlist(self, boxes, scores, image_shape, label_ids):
        """
        Converts the boxes, represented as (N, num_classes * 4) tensor
        into a BoxList containing N * num_classes boxes.
        Additionally, it also adds probability scores and class information
        to each box as extra fields
        """
        num_locations = boxes.shape[0]
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        # labels = label_ids.repeat(num_locations)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        # boxlist.add_field("labels", labels)
        return boxlist

    def filter_results(self, boxlist, num_classes):
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        cls_scores, cls_boxes, labels = box_results_with_nms_and_limit(
            scores,
            boxes,
            self.score_thresh,
            self.nms,
            self.detections_per_img,
        )
        bbox = BoxList(cls_boxes, boxlist.size, mode="xyxy")
        bbox.add_field("scores", cls_scores)
        bbox.add_field("labels", labels)
        return bbox

    def filter_results_0(self, boxlist, num_classes):
        scores = boxlist.get_field("scores")
        inds = (scores > self.score_thresh).nonzero().squeeze(1)
        boxlist = boxlist[inds]

        labels = boxlist.get_field("labels")

        result = []
        for cls in range(1, num_classes):
            idx = (labels == cls).nonzero().squeeze(1)
            boxlist_for_class = boxlist[idx]
            boxlist_for_class = boxlist_nms(boxlist_for_class, self.nms,
                    score_field="scores")
            result.append(boxlist_for_class)

        result = cat_bbox(result)
        number_of_detections = len(result)
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result

