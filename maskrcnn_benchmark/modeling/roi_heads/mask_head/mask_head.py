import torch
from torch.nn import functional as F

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.utils import cat

from .roi_mask_feature_extractors import make_roi_mask_feature_extractor
from .roi_mask_predictors import make_roi_mask_predictor
from .inference import make_roi_mask_post_processor


def project_masks_on_boxes(segmentation_masks, proposals, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )
    # TODO put the proposals on the CPU, as the representation for the
    # masks is not efficient GPU-wise (possibly several small tensors for
    # representing a single instance mask)
    proposals = proposals.bbox.to(torch.device("cpu"))
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation,
        # instead of the list representation that was used
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.convert(mode="mask")
        masks.append(mask)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


def maskrcnn_loss(proposals, mask_logits, matched_targets, discretization_size):
    """
    Arguments:
        proposals (list[BoxList]): all the foreground proposals
        mask_logits (Tensor)
        matched_targets (list[BoxList]): one-to-one corresponds to the proposals.

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """
    labels = []
    mask_targets = []
    for proposals_per_image, matched_targets_per_image in zip(proposals, matched_targets):
        labels_per_image = matched_targets_per_image.get_field("labels").to(dtype=torch.int64)
        segmentation_masks = matched_targets_per_image.get_field("masks")
        masks_per_image = project_masks_on_boxes(
            segmentation_masks, proposals_per_image, discretization_size
        )

        labels.append(labels_per_image)
        mask_targets.append(masks_per_image)

    labels = cat(labels, dim=0)
    mask_targets = cat(mask_targets, dim=0)

    positive_inds = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[positive_inds]

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits[positive_inds, labels_pos], mask_targets
    )
    return mask_loss


def keep_only_positive_boxes(boxes, matched_targets):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list[BoxList])
        matched_targets (list[BoxList]):
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
        self.discretization_size = cfg.MODEL.ROI_MASK_HEAD.RESOLUTION

    def forward(self, features, proposals, matched_targets=None):
        """
        Args:
            proposals (list[BoxList]):
            matched_targets (list[BoxList]):
                one-to-one corresponds to the proposals. Only those corresponds to foreground proposals are meaningful.
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

        if self.training:
            loss_mask = maskrcnn_loss(proposals, mask_logits, matched_targets, self.discretization_size)
            return x, all_proposals, dict(loss_mask=loss_mask)
        else:
            result = self.post_processor(mask_logits, proposals)
            return x, result, {}


def build_roi_mask_head(cfg):
    return ROIMaskHead(cfg)
