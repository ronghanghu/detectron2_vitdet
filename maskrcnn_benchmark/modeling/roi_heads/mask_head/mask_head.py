import torch

from maskrcnn_benchmark.structures.bounding_box import BoxList

from .roi_mask_feature_extractors import make_roi_mask_feature_extractor
from .roi_mask_predictors import make_roi_mask_predictor
from .inference import make_roi_mask_post_processor
from .loss import make_roi_mask_loss_evaluator


def keep_only_positive_boxes(boxes, matched_targets):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    positive_targets = []
    for boxes_per_image, targets_per_image in zip(boxes, matched_targets):
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
        positive_targets.append(targets_per_image[inds])
    return positive_boxes, positive_inds, positive_targets


class ROIMaskHead(torch.nn.Module):
    def __init__(self, cfg):
        super(ROIMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_mask_feature_extractor(cfg)
        self.predictor = make_roi_mask_predictor(cfg)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)

    def forward(self, features, proposals, matched_targets=None):
        """
        Args:
            proposals (list[BoxList]):
            matched_targets (list[BoxList]): one-to-one corresponds to the proposals
        """
        if self.training:
            # during training, only focus on positive boxes
            assert len(matched_targets) == len(proposals)
            assert len(matched_targets[0]) == len(proposals[0])
            all_proposals = proposals
            proposals, positive_inds, matched_targets = keep_only_positive_boxes(proposals, matched_targets)

        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features, proposals)
        mask_logits = self.predictor(x)

        if not self.training:
            result = self.post_processor(mask_logits, proposals)
            return x, result, {}

        loss_mask = self.loss_evaluator(proposals, mask_logits, matched_targets)

        return x, all_proposals, dict(loss_mask=loss_mask)


def build_roi_mask_head(cfg):
    return ROIMaskHead(cfg)
