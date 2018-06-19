import torch
from torch import nn
import torch.nn.functional as F

from ..structures.bounding_box import BBox

from .box_coder import BoxCoder
from .box_selector import _clip_boxes_to_image

from torchvision.layers import nms as box_nms


def box_results_with_nms_and_limit(scores, boxes, score_thresh=0.05, nms=0.5, detections_per_img=100):
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
    cls_boxes = [[] for _ in range(num_classes)]
    cls_scores = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = scores[:, j] > score_thresh
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4:(j + 1) * 4]
        keep = box_nms(boxes_j, scores_j, nms)
        cls_boxes[j] = boxes_j[keep]
        cls_scores[j] = scores_j[keep]
        labels[j] = torch.full_like(keep, j)

    cls_scores = [s for s in cls_scores if len(s) > 0]
    cls_boxes = [s for s in cls_boxes if len(s) > 0]
    labels = [s for s in labels if len(s) > 0]

    if len(cls_scores) > 0:
        cls_scores = torch.cat(cls_scores, dim=0)
        cls_boxes = torch.cat(cls_boxes, dim=0)
        labels = torch.cat(labels, dim=0)
    else:
        device = scores.device
        cls_scores = scores.new()
        cls_boxes = boxes.new()
        labels = torch.empty(0, dtype=torch.int64, device=device)

    number_of_detections = len(cls_scores)

    # Limit to max_per_image detections **over all classes**
    if number_of_detections > detections_per_img > 0:
        image_thresh, _ = torch.kthvalue(cls_scores.cpu(),
                number_of_detections - detections_per_img + 1)
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
    def __init__(self, score_thresh=0.05, nms=0.5, detections_per_img=100,
            box_coder=BoxCoder(weights=(10., 10., 5., 5.))):
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
            x (tuple[tensor, tensor])
            boxes (list[list[BBox]])
        """
        assert len(boxes) == 1, 'Only single feature'
        boxes = boxes[0]
        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size[::-1] for box in boxes]
        boxes_per_image = [box.bbox.size(0) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        proposals = self.box_coder.decode(
                box_regression.view(sum(boxes_per_image), -1), concat_boxes)

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, proposal, (height, width) in zip(class_prob, proposals, image_shapes):
            clipped_proposal = _clip_boxes_to_image(proposal, height, width)
            cls_scores, cls_boxes, labels = box_results_with_nms_and_limit(
                    prob, clipped_proposal, self.score_thresh, self.nms,
                    self.detections_per_img)
            bbox = BBox(cls_boxes, (width, height), mode='xyxy')
            bbox.add_field('scores', cls_scores)
            bbox.add_field('labels', labels)
            results.append(bbox)

        return results


# TODO maybe simplify
class FPNPostProcessor(nn.Module):
    def __init__(self, score_thresh=0.05, nms=0.5, detections_per_img=100,
            box_coder=BoxCoder(weights=(10., 10., 5., 5.))):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(FPNPostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        self.box_coder = box_coder

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor])
            boxes (list[list[BBox]])
        """
        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)

        num_features = len(boxes)
        num_images = len(boxes[0])

        merged_lists = [box for per_feature_boxes in boxes for box in per_feature_boxes]
        indices = [torch.full((box.bbox.shape[0],), img_idx) for per_feature_boxes in boxes
                for img_idx, box in enumerate(per_feature_boxes)]

        # TODO make it a helper function
        concat_boxes = torch.cat([b.bbox for b in merged_lists], dim=0)
        indices = torch.cat(indices, dim=0)
        extra_fields = {}
        field_names = merged_lists[0].fields()
        for field in field_names:
            extra_fields[field] = torch.cat([b.get_field(field) for b in merged_lists], dim=0)

        image_shapes = [box.size[::-1] for box in boxes[0]]

        proposals = self.box_coder.decode(box_regression, concat_boxes)

        all_proposals = []
        all_class_prob = []
        for img_idx in range(num_images):
            img_mask = indices == img_idx
            all_proposals.append(proposals[img_mask])
            all_class_prob.append(class_prob[img_mask])
        proposals = all_proposals
        class_prob = all_class_prob

        results = []
        for prob, proposal, (height, width) in zip(class_prob, proposals, image_shapes):
            clipped_proposal = _clip_boxes_to_image(proposal, height, width)
            cls_scores, cls_boxes, labels = box_results_with_nms_and_limit(
                    prob, clipped_proposal, self.score_thresh, self.nms,
                    self.detections_per_img)
            bbox = BBox(cls_boxes, (width, height), mode='xyxy')
            bbox.add_field('scores', cls_scores)
            bbox.add_field('labels', labels)
            results.append(bbox)

        return results
