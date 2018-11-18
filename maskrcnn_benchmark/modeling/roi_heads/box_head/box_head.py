import torch
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms, cat_boxlist

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor


def fastrcnn_losses(labels, regression_targets, class_logits, regression_outputs):
    """
    Computes the box classification & regression loss for Faster R-CNN.

    Arguments:
        labels (Tensor): #box labels. Each of range [0, #class].
        class_logits (Tensor): #box x #class
        box_regression (Tensor): #box x (#class x 4)

    Returns:
        classification_loss, regression_loss
    """
    device = class_logits.device

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)

    box_loss = smooth_l1_loss(
        regression_outputs[sampled_pos_inds_subset[:, None], map_inds],
        regression_targets[sampled_pos_inds_subset],
        size_average=False,
        beta=1,
    )
    box_loss = box_loss / labels.numel()
    return classification_loss, box_loss


def fastrcnn_inference(
        boxes, scores, image_shape,
        score_thresh, nms_thresh, detections_per_img):
    """
    Per-image inference post-processing function.
    Returns bounding-box detection results by thresholding on scores
    and applying non-maximum suppression (NMS).

    Args:
        boxes (Tensor): N, #class, 4
        scores (Tensor): N, #class

    Returns:
        BoxList:
    """
    num_classes = scores.shape[1]
    # convert to boxlist to use the `clip` function ...
    boxlist = BoxList(boxes.reshape(-1, 4),
                      image_shape, mode="xyxy")
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
        boxlist_for_class = boxlist_nms(
            boxlist_for_class, nms_thresh, score_field="scores"
        )
        num_labels = len(boxlist_for_class)
        boxlist_for_class.add_field(
            "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
        )
        result.append(boxlist_for_class)

    result = cat_boxlist(result)
    number_of_detections = len(result)

    # Limit to max_per_image detections **over all classes**
    if number_of_detections > detections_per_img > 0:
        cls_scores = result.get_field("scores")
        image_thresh, _ = torch.kthvalue(
            cls_scores.cpu(), number_of_detections - detections_per_img + 1
        )
        keep = cls_scores >= image_thresh.item()
        keep = torch.nonzero(keep).squeeze(1)
        result = result[keep]
    return result


class FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.

    TODO more property/methods can be added,
    e.g. class-agnostic regression, predicted_labels, decoded_boxes_for_gt_labels, fg_proposals
    """
    def __init__(self, box_coder, class_logits, regression_outputs, proposals, targets=None):
        """
        Args:
            class_logits (Tensor): #box x #class
            regression_outputs (Tensor): #box x #class x 4
            proposals (list[BoxList]): has a field "labels" if training
            targets (list[BoxList]): None if testing
        """
        self.box_coder = box_coder
        self.num_img = len(proposals)
        self.img_shapes = [box.size for box in proposals]
        self.num_boxes = [len(box) for box in proposals]
        self.num_classes = class_logits.shape[1]

        self.proposals = [box.bbox for box in proposals]
        if proposals[0].has_field("labels"):
            self.labels = [box.get_field("labels") for box in proposals]

        self.class_logits = class_logits
        self.regression_outputs = regression_outputs
        self.targets = targets

    def losses(self):
        """
        Returns:
            dict
        """
        regression_targets = [
            self.box_coder.encode(targets_per_image.bbox, proposals_per_image)
            for targets_per_image, proposals_per_image in zip(self.targets, self.proposals)]

        loss_classifier, loss_box_reg = fastrcnn_losses(
            cat(self.labels, dim=0),
            cat(regression_targets, dim=0),
            self.class_logits, self.regression_outputs)
        return dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)

    def decoded_outputs(self):
        """
        Returns:
            list[Tensor]: the Tensor for each image has shape (#box, #class, 4)
        """
        decoded = self.box_coder.decode(
            self.regression_outputs.view(sum(self.num_boxes), -1),
            torch.cat(self.proposals, dim=0))
        return decoded.split(self.num_boxes, dim=0)

    def predicted_probs(self):
        """
        Returns:
            list[Tensor]: the Tensor for each image has shape (#box, #class)
        """
        probs = F.softmax(self.class_logits, -1)
        return probs.split(self.num_boxes, dim=0)


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(ROIBoxHead, self).__init__()
        self.cfg = cfg.MODEL.ROI_HEADS.clone()
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        self.predictor = make_roi_box_predictor(cfg)
        bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
        self.box_coder = BoxCoder(weights=bbox_reg_weights)

    def forward(self, features, proposals, matched_targets=None):
        x = self.feature_extractor(features, proposals)
        class_logits, regression_outputs = self.predictor(x)
        # #box x #class, #box x #class x 4

        head_outputs = FastRCNNOutputs(
            self.box_coder, class_logits, regression_outputs, proposals, matched_targets)
        if not self.training:
            decoded_boxes = head_outputs.decoded_outputs()
            probs = head_outputs.predicted_probs()

            results = []
            for probs_per_image, boxes_per_image, image_shape in zip(probs, decoded_boxes, head_outputs.img_shapes):
                boxlist = fastrcnn_inference(
                    boxes_per_image, probs_per_image, image_shape,
                    self.cfg.SCORE_THRESH, self.cfg.NMS, self.cfg.DETECTIONS_PER_IMG)
                results.append(boxlist)
            return x, results, {}
        else:
            return x, proposals, head_outputs.losses()


def build_roi_box_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg)
