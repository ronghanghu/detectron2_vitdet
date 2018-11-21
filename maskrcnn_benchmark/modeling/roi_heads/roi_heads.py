import torch
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    sample_with_positive_fraction,
)
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

from .box_head import FastRCNNOutputHead
from .box_head import FastRCNNOutputs
from .box_head import fastrcnn_inference
from .box_head import make_box_head
from .mask_head import make_mask_head
from .mask_head import maskrcnn_inference
from .mask_head import maskrcnn_loss


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
    positive_masks = []
    positive_targets = []
    for boxes_per_image, targets_per_image in zip(boxes, matched_targets):
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_targets.append(targets_per_image[inds])
        positive_masks.append(inds_mask)
    return positive_boxes, positive_targets, positive_masks


class ROIHeads(torch.nn.Module):
    def __init__(self, cfg):
        super(ROIHeads, self).__init__()
        self.cfg = cfg.clone()

        # match proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
            cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
            allow_low_quality_matches=False,
        )
        self.batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION

        self.test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
        self.test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS
        self.test_detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG

    def sample_proposals_for_training(self, proposals, targets):
        """
        Sample the proposals and prepare their training targets.

        Args:
            proposals (list[BoxList]): #img BoxList. Each contains the proposals for the image.
            targets (list[BoxList]): #img BoxList. The GT boxes for the image.

        Returns:
            list[BoxList]: The proposals after sampling. It has a "labels" field, ranging in [0, #class]
            list[BoxList]: The matched targets for each sampled proposal.
                Only those for foreground proposals are meaningful.
        """
        sampled_targets = []

        with torch.no_grad():
            for image_idx, (proposals_per_image, targets_per_image) in enumerate(
                zip(proposals, targets)
            ):
                match_quality_matrix = boxlist_iou(targets_per_image, proposals_per_image)
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # Fast RCNN only need "labels" field for selecting the targets
                target = targets_per_image.copy_with_fields(["labels", "masks"])
                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                matched_targets = target[matched_idxs.clamp(min=0)]

                labels_per_image = matched_targets.get_field("labels").to(dtype=torch.int64)

                # Label background (below the low threshold)
                labels_per_image[matched_idxs == Matcher.BELOW_LOW_THRESHOLD] = 0
                # Label ignore proposals (between low and high thresholds)
                labels_per_image[
                    matched_idxs == Matcher.BETWEEN_THRESHOLDS
                ] = -1  # -1 is ignored by sampler

                # apply sampling
                sampled_pos_inds, sampled_neg_inds = sample_with_positive_fraction(
                    labels_per_image, self.batch_size_per_image, self.positive_sample_fraction
                )
                sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

                sampled_targets.append(matched_targets[sampled_inds])
                proposals[image_idx] = proposals_per_image[sampled_inds]
                proposals[image_idx].add_field("labels", labels_per_image[sampled_inds])
        return proposals, sampled_targets

    def forward(self, features, proposals, targets=None):
        raise NotImplementedError()


class C4ROIHeads(ROIHeads):
    def __init__(self, cfg):
        super(C4ROIHeads, self).__init__(cfg)

        from maskrcnn_benchmark.modeling.backbone import resnet

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        self.pooler = Pooler(
            output_size=resolution,
            scales=cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES,
            sampling_ratio=cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO,
        )

        self.res5_head = resnet.ResNetHead(
            block_module=cfg.MODEL.RESNETS.TRANS_FUNC,
            stages=(resnet.StageSpec(index=4, block_count=3, return_features=False),),
            num_groups=cfg.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=cfg.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=cfg.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=cfg.MODEL.RESNETS.RES2_OUT_CHANNELS,
        )

        num_channels = self.res5_head.output_size
        self.box_predictor = FastRCNNOutputHead(num_channels, cfg.MODEL.ROI_HEADS.NUM_CLASSES)
        self.box_coder = BoxCoder(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

        if cfg.MODEL.MASK_ON:
            self.mask_head = make_mask_head(cfg, (num_channels, resolution, resolution))
            self.mask_discretization_size = cfg.MODEL.ROI_MASK_HEAD.RESOLUTION

    def _shared_roi_transform(self, features, proposals):
        x = self.pooler(features, proposals)
        return self.res5_head(x)

    def forward(self, features, proposals, targets=None):
        if self.training:
            proposals, targets = self.sample_proposals_for_training(proposals, targets)

        box_features = self._shared_roi_transform(features, proposals)
        feature_pooled = F.avg_pool2d(box_features, 7)  # gap
        class_logits, regression_outputs = self.box_predictor(feature_pooled)
        del feature_pooled

        outputs = FastRCNNOutputs(
            self.box_coder, class_logits, regression_outputs, proposals, targets
        )

        if self.training:
            del features
            losses = outputs.losses()

            if self.cfg.MODEL.MASK_ON:
                proposals, targets, pos_masks = keep_only_positive_boxes(proposals, targets)
                # don't need to do feature extraction again,
                # just use the box features for all the positivie proposals
                mask_features = box_features[torch.cat(pos_masks, dim=0)]
                del box_features
                mask_logits = self.mask_head(mask_features)
                losses["loss_mask"] = maskrcnn_loss(
                    proposals, mask_logits, targets, self.mask_discretization_size
                )
            return None, None, losses
        else:
            results = fastrcnn_inference(
                outputs.predict_boxes(),
                outputs.predict_probs(),
                outputs.image_shapes,
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            if self.cfg.MODEL.MASK_ON:
                x = self._shared_roi_transform(features, results)
                mask_logits = self.mask_head(x)
                maskrcnn_inference(mask_logits, results)
            return None, results, {}


class StandardROIHeads(ROIHeads):
    """
    Standard in a sense that no ROI transform sharing or feature sharing.
    The rois go to separate branches (boxes and masks) directly.
    This is used by FPN, C5 models, etc.
    """

    def __init__(self, cfg):
        super(StandardROIHeads, self).__init__(cfg)

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        self.box_pooler = Pooler(
            output_size=resolution,
            scales=cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES,
            sampling_ratio=cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO,
        )
        self.box_head = make_box_head(
            cfg, (cfg.MODEL.BACKBONE.OUT_CHANNELS, resolution, resolution)
        )

        self.box_predictor = FastRCNNOutputHead(
            self.box_head.output_size, cfg.MODEL.ROI_HEADS.NUM_CLASSES
        )
        self.box_coder = BoxCoder(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

        if cfg.MODEL.MASK_ON:
            self.mask_pooler = Pooler(
                output_size=cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION,
                scales=cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES,
                sampling_ratio=cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO,
            )
            self.mask_head = make_mask_head(cfg, cfg.MODEL.BACKBONE.OUT_CHANNELS)
            self.mask_discretization_size = cfg.MODEL.ROI_MASK_HEAD.RESOLUTION

    def forward(self, features, proposals, targets=None):
        if self.training:
            proposals, targets = self.sample_proposals_for_training(proposals, targets)

        box_features = self.box_pooler(features, proposals)
        box_features = self.box_head(box_features)
        class_logits, regression_outputs = self.box_predictor(box_features)
        del box_features

        outputs = FastRCNNOutputs(
            self.box_coder, class_logits, regression_outputs, proposals, targets
        )

        if self.training:
            losses = outputs.losses()
        else:
            losses = {}
            proposals = fastrcnn_inference(
                outputs.predict_boxes(),
                outputs.predict_probs(),
                outputs.image_shapes,
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )

        if self.cfg.MODEL.MASK_ON:
            if self.training:
                proposals, targets, _ = keep_only_positive_boxes(proposals, targets)
            mask_features = self.mask_pooler(features, proposals)
            mask_logits = self.mask_head(mask_features)

            if self.training:
                losses["loss_mask"] = maskrcnn_loss(
                    proposals, mask_logits, targets, self.mask_discretization_size
                )
            else:
                maskrcnn_inference(mask_logits, proposals)

        return None, proposals, losses


def build_roi_heads(cfg):
    name = cfg.MODEL.ROI_HEADS.NAME
    return {"C4ROIHeads": C4ROIHeads, "StandardROIHeads": StandardROIHeads}[name](cfg)
