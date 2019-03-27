import numpy as np
import torch
from torch.nn import functional as F

from detectron2.structures import Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from ..backbone import resnet
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..poolers import ROIPooler
from ..sampling import subsample_labels
from .box_head import build_box_head
from .fast_rcnn import FastRCNNOutputHead, FastRCNNOutputs
from .keypoint_head import build_keypoint_head, keypoint_rcnn_inference, keypoint_rcnn_loss
from .mask_head import build_mask_head, mask_rcnn_inference, mask_rcnn_loss

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")


def build_roi_heads(cfg):
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg)


def select_foreground_proposals(proposals):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only boxes with `gt_classes > 0`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = gt_classes > 0
        fg_inds = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_inds])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


class ROIHeads(torch.nn.Module):
    def __init__(self, cfg):
        super(ROIHeads, self).__init__()

        # match proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
            cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
            allow_low_quality_matches=False,
        )

        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS
        self.test_detections_per_img  = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.feature_strides          = dict(cfg.MODEL.BACKBONE.COMPUTED_OUT_FEATURE_STRIDES)
        self.feature_channels         = dict(cfg.MODEL.BACKBONE.COMPUTED_OUT_FEATURE_CHANNELS)
        # fmt: on

    def label_and_sample_proposals(self, proposals, targets):
        """
        Perform box matching between `proposals` and `targets`, and label the
        proposals with training labels. Return `self.batch_size_per_image`
        random samples from `proposals` with a fraction of positives that is no
        larger than `self.positive_sample_fraction.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]: length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        with torch.no_grad():
            for proposals_per_image, targets_per_image in zip(proposals, targets):
                match_quality_matrix = pairwise_iou(
                    targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
                )
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # Fast RCNN only need "gt_classes" field for selecting the targets
                # Get the targets corresponding GT for each proposal
                # Need to clamp the indices because matched_idxs can be < 0
                matched_idxs_clamped = matched_idxs.clamp(min=0)

                gt_classes = targets_per_image.gt_classes[matched_idxs_clamped].to(
                    dtype=torch.int64
                )

                # Label background (below the low threshold)
                gt_classes[matched_idxs == Matcher.BELOW_LOW_THRESHOLD] = 0
                # Label ignore proposals (between low and high thresholds)
                gt_classes[matched_idxs == Matcher.BETWEEN_THRESHOLDS] = -1

                sampled_fg_inds, sampled_bg_inds = subsample_labels(
                    gt_classes, self.batch_size_per_image, self.positive_sample_fraction
                )

                num_fg_samples.append(sampled_fg_inds.numel())
                num_bg_samples.append(sampled_bg_inds.numel())

                sampled_inds = torch.cat([sampled_fg_inds, sampled_bg_inds], dim=0)

                proposals_per_image = proposals_per_image[sampled_inds]
                proposals_per_image.gt_classes = gt_classes[sampled_inds]

                # Avoid indexing the Boxes targets_per_image directly, as in the
                # commented-out row below. It is considerably slower.
                # TODO: investigate and optimize this access pattern.
                # gt_boxes = targets_per_image[matched_idxs_clamped[sampled_inds]].bbox  # slow!
                gt_boxes = targets_per_image.gt_boxes[matched_idxs_clamped[sampled_inds]]
                proposals_per_image.gt_boxes = gt_boxes

                if targets_per_image.has("gt_masks"):
                    # See note above about not indexing the targets_per_image directly
                    gt_masks = targets_per_image.gt_masks[matched_idxs_clamped[sampled_inds]]
                    proposals_per_image.gt_masks = gt_masks
                if targets_per_image.has("gt_keypoints"):
                    gt_keypoints = targets_per_image.gt_keypoints[
                        matched_idxs_clamped[sampled_inds]
                    ]
                    proposals_per_image.gt_keypoints = gt_keypoints

                proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(self, images, features, proposals, targets=None):
        """
        Args:
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:
                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: the ground-truth mask of the instance.

        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
                detected instances. Returned during inference only; may be []
                during training.
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    def __init__(self, cfg):
        super(Res5ROIHeads, self).__init__(cfg)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution   = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales       = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio      = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        num_classes         = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        bbox_reg_weights    = cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
        self.mask_side_len  = cfg.MODEL.ROI_MASK_HEAD.RESOLUTION
        self.mask_on        = cfg.MODEL.MASK_ON
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution, scales=pooler_scales, sampling_ratio=sampling_ratio
        )

        self.res5 = resnet.build_resnet_head(cfg)
        out_channels = self.res5[-1].out_channels
        self.box_predictor = FastRCNNOutputHead(out_channels, num_classes)
        self.box2box_transform = Box2BoxTransform(weights=bbox_reg_weights)

        if self.mask_on:
            cfg = cfg.clone()
            cfg.MODEL.ROI_MASK_HEAD.COMPUTED_INPUT_SIZE = (
                out_channels,
                pooler_resolution,
                pooler_resolution,
            )
            self.mask_head = build_mask_head(cfg)

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images

        features = [features[f] for f in self.in_features]
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(features, proposal_boxes)
        feature_pooled = F.avg_pool2d(box_features, 7)  # gap
        pred_class_logits, pred_proposal_deltas = self.box_predictor(feature_pooled)
        del feature_pooled

        outputs = FastRCNNOutputs(
            self.box2box_transform, pred_class_logits, pred_proposal_deltas, proposals
        )

        if self.training:
            del features
            losses = outputs.losses()
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(proposals)
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                mask_logits = self.mask_head(mask_features)
                losses["loss_mask"] = mask_rcnn_loss(mask_logits, proposals, self.mask_side_len)
            return [], losses
        else:
            pred_instances = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            if self.mask_on:
                x = self._shared_roi_transform(features, [x.pred_boxes for x in pred_instances])
                mask_logits = self.mask_head(x)
                mask_rcnn_inference(mask_logits, pred_instances)
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    Standard in a sense that no ROI transform sharing or feature sharing.
    The rois go to separate branches (boxes and masks) directly.
    This is used by FPN, C5 models, etc.
    """

    def __init__(self, cfg):
        super(StandardROIHeads, self).__init__(cfg)

        # fmt: off
        pooler_resolution                        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales                            = tuple(1.0 / self.feature_strides[k] for k in self.in_features)  # noqa
        sampling_ratio                           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        num_classes                              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        bbox_reg_weights                         = cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
        self.mask_on                             = cfg.MODEL.MASK_ON
        self.mask_side_len                       = cfg.MODEL.ROI_MASK_HEAD.RESOLUTION
        mask_pooler_resolution                   = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        mask_pooler_scales                       = pooler_scales
        mask_sampling_ratio                      = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        self.keypoint_on                         = cfg.MODEL.KEYPOINT_ON
        keypoint_pooler_resolution               = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        keypoint_pooler_scales                   = pooler_scales
        keypoint_sampling_ratio                  = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        self.normalize_loss_by_visible_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS  # noqa
        self.keypoint_loss_weight                = cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        cfg = cfg.clone()
        self.box_pooler = ROIPooler(
            output_size=pooler_resolution, scales=pooler_scales, sampling_ratio=sampling_ratio
        )
        cfg.MODEL.ROI_BOX_HEAD.COMPUTED_INPUT_SIZE = (
            in_channels,
            pooler_resolution,
            pooler_resolution,
        )
        self.box_head = build_box_head(cfg)

        self.box_predictor = FastRCNNOutputHead(self.box_head.output_size, num_classes)
        self.box2box_transform = Box2BoxTransform(weights=bbox_reg_weights)

        if self.mask_on:
            self.mask_pooler = ROIPooler(
                output_size=mask_pooler_resolution,
                scales=mask_pooler_scales,
                sampling_ratio=mask_sampling_ratio,
            )
            cfg.MODEL.ROI_MASK_HEAD.COMPUTED_INPUT_SIZE = (
                in_channels,
                mask_pooler_resolution,
                mask_pooler_resolution,
            )
            self.mask_head = build_mask_head(cfg)

        if self.keypoint_on:
            self.keypoint_pooler = ROIPooler(
                output_size=keypoint_pooler_resolution,
                scales=keypoint_pooler_scales,
                sampling_ratio=keypoint_sampling_ratio,
            )
            cfg.MODEL.ROI_KEYPOINT_HEAD.COMPUTED_INPUT_SIZE = (
                in_channels,
                keypoint_pooler_resolution,
                keypoint_pooler_resolution,
            )
            self.keypoint_head = build_keypoint_head(cfg)

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        num_images = len(images)
        del images

        features = [features[f] for f in self.in_features]
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform, pred_class_logits, pred_proposal_deltas, proposals
        )

        if self.training:
            losses = outputs.losses()
            if self.mask_on or self.keypoint_on:
                proposals, _ = select_foreground_proposals(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            # During training the same proposals used by the box head are used by
            # the mask and keypoint heads. The loss is only defined on positive proposals.
            if self.mask_on:
                mask_features = self.mask_pooler(features, proposal_boxes)
                mask_logits = self.mask_head(mask_features)
                losses["loss_mask"] = mask_rcnn_loss(mask_logits, proposals, self.mask_side_len)
            if self.keypoint_on:
                keypoint_features = self.keypoint_pooler(features, proposal_boxes)
                keypoint_logits = self.keypoint_head(keypoint_features)

                normalizer = (
                    num_images
                    * self.batch_size_per_image
                    * self.positive_sample_fraction
                    * keypoint_logits.shape[1]
                )
                losses["loss_keypoint"] = keypoint_rcnn_loss(
                    keypoint_logits,
                    proposals,
                    normalizer=None if self.normalize_loss_by_visible_keypoints else normalizer,
                )
                losses["loss_keypoint"] *= self.keypoint_loss_weight
            return [], losses
        else:
            pred_instances = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_boxes = [x.pred_boxes for x in pred_instances]
            if self.mask_on:
                # During inference cascaded prediction is used: the mask head is only
                # applied to the top scoring box detections.
                mask_features = self.mask_pooler(features, pred_boxes)
                mask_logits = self.mask_head(mask_features)
                mask_rcnn_inference(mask_logits, pred_instances)
            if self.keypoint_on:
                keypoint_features = self.keypoint_pooler(features, pred_boxes)
                keypoint_logits = self.keypoint_head(keypoint_features)
                keypoint_rcnn_inference(keypoint_logits, pred_instances)
            return pred_instances, {}
