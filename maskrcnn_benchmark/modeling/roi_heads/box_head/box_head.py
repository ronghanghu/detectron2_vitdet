import torch
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor


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


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        self.predictor = make_roi_box_predictor(cfg)
        self.post_processor = make_roi_box_post_processor(cfg)

        bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
        self.box_coder = BoxCoder(weights=bbox_reg_weights)


    def forward(self, features, proposals, labels=None, matched_targets=None):
        x = self.feature_extractor(features, proposals)
        class_logits, regression_outputs = self.predictor(x)
        # #box x #class, #box x #class x 4

        if not self.training:
            result = self.post_processor((class_logits, regression_outputs), proposals)
            return x, result, {}
        else:
            regression_targets = [
                self.box_coder.encode(matched_targets_per_image.bbox, proposals_per_image.bbox)
                for matched_targets_per_image, proposals_per_image in zip(matched_targets, proposals)]

            loss_classifier, loss_box_reg = fastrcnn_losses(
                cat(labels, dim=0),
                cat(regression_targets, dim=0),
                class_logits, regression_outputs)
            return (
                x,
                proposals,
                dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
            )


def build_roi_box_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg)
