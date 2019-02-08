import torch
from torch.nn import functional as F

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.utils.registry import Registry

from ..backbone import resnet
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..poolers import ROIPooler
from ..sampling import subsample_labels
from .box_head import build_box_head
from .fast_rcnn import FastRCNNOutputHead, FastRCNNOutputs, fastrcnn_inference
from .mask_head import build_mask_head, maskrcnn_inference, maskrcnn_loss


ROI_HEADS_REGISTRY = Registry("ROI_HEADS")


def build_roi_heads(cfg):
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg)


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
        self.feature_strides          = dict(cfg.MODEL.BACKBONE.OUT_FEATURE_STRIDES)
        self.feature_channels         = dict(cfg.MODEL.BACKBONE.OUT_FEATURE_CHANNELS)
        # fmt: on

    def sample_proposals_for_training(self, proposals, targets):
        """
        Perform box matching between `proposals` and `targets`, and label the
        proposals as positive/negative/ignored. Return `self.batch_size_per_image`
        random samples from `proposals` with a fraction of positives that is no
        larger than `self.positive_sample_fraction.

        Args:
            proposals (list[BoxList]): length `N` (number of images) list of
                `BoxList`s. The i-th `BoxList` contains object proposals for the
                i-th input image
            targets (list[BoxList]): length `N` (number of images) list of
                `BoxList`s. The i-th `BoxList` contains the ground-truth boxes
                for the i-th input image.

        Returns:
            list[BoxList]: length `N` list of `BoxList`s containing the proposals
                sampled for training. Each `BoxList` has a "labels" field, labeling
                each box with a category ranging in [0, #class].
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
                # Get the targets corresponding GT for each proposal
                # Need to clamp the indices because matched_idxs can be < 0
                matched_idxs_clamped = matched_idxs.clamp(min=0)

                labels_per_image = targets_per_image.get_field("labels")[matched_idxs_clamped].to(
                    dtype=torch.int64
                )

                # Label background (below the low threshold)
                labels_per_image[matched_idxs == Matcher.BELOW_LOW_THRESHOLD] = 0
                # Label ignore proposals (between low and high thresholds)
                labels_per_image[
                    matched_idxs == Matcher.BETWEEN_THRESHOLDS
                ] = -1  # -1 is ignored by sampler

                # apply sampling
                sampled_pos_inds, sampled_neg_inds = subsample_labels(
                    labels_per_image, self.batch_size_per_image, self.positive_sample_fraction
                )
                sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

                # It's still unnecessary to index the masks here.
                # Because for masks we only need the positives.
                # This will have another tiny (2~3%) perf improvement.
                sampled_targets.append(targets_per_image[matched_idxs_clamped[sampled_inds]])
                proposals[image_idx] = proposals_per_image[sampled_inds]
                proposals[image_idx].add_field("labels", labels_per_image[sampled_inds])
        return proposals, sampled_targets

    def forward(self, features, proposals, targets=None):
        """
        Args:
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[BoxList]): length `N` list of `BoxList`s. The i-th
                `BoxList` contains object proposals for the i-th input image
            targets (list[BoxList], optional): length `N` list of `BoxList`s. The i-th
                `BoxList` contains the ground-truth boxes for the i-th input image.
                Specify `targets` during training only.

        Returns:
            results (list[BoxList]): length `N` list of `BoxList`s containing the
                detected objects. Returned during inference only; may be None
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
        pooler_resolution             = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales                 = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio                = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        num_classes                   = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        bbox_reg_weights              = cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
        self.mask_discretization_size = cfg.MODEL.ROI_MASK_HEAD.RESOLUTION
        self.mask_on                  = cfg.MODEL.MASK_ON
        # fmt: on

        self.pooler = ROIPooler(
            output_size=pooler_resolution, scales=pooler_scales, sampling_ratio=sampling_ratio
        )

        self.res5 = resnet.build_resnet_head(cfg)
        num_channels = self.res5[-1].out_channels
        self.box_predictor = FastRCNNOutputHead(num_channels, num_classes)
        self.box2box_transform = Box2BoxTransform(weights=bbox_reg_weights)

        if self.mask_on:
            self.mask_head = build_mask_head(
                cfg, (num_channels, pooler_resolution, pooler_resolution)
            )

    def _shared_roi_transform(self, features, proposals):
        x = self.pooler(features, proposals)
        return self.res5(x)

    def forward(self, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        features = [features[f] for f in self.in_features]
        if self.training:
            proposals, targets = self.sample_proposals_for_training(proposals, targets)

        box_features = self._shared_roi_transform(features, proposals)
        feature_pooled = F.avg_pool2d(box_features, 7)  # gap
        class_logits, regression_outputs = self.box_predictor(feature_pooled)
        del feature_pooled

        outputs = FastRCNNOutputs(
            self.box2box_transform, class_logits, regression_outputs, proposals, targets
        )

        if self.training:
            del features
            losses = outputs.losses()
            if self.mask_on:
                proposals, targets, pos_masks = keep_only_positive_boxes(proposals, targets)
                # don't need to do feature extraction again,
                # just use the box features for all the positivie proposals
                mask_features = box_features[torch.cat(pos_masks, dim=0)]
                del box_features
                mask_logits = self.mask_head(mask_features)
                losses["loss_mask"] = maskrcnn_loss(
                    proposals, mask_logits, targets, self.mask_discretization_size
                )
            return None, losses
        else:
            results = fastrcnn_inference(
                outputs.predict_boxes(),
                outputs.predict_probs(),
                outputs.image_shapes,
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            if self.mask_on:
                x = self._shared_roi_transform(features, results)
                mask_logits = self.mask_head(x)
                maskrcnn_inference(mask_logits, results)
            return results, {}


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
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        bbox_reg_weights         = cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
        self.mask_on             = cfg.MODEL.MASK_ON
        if self.mask_on:
            self.mask_discretization_size = cfg.MODEL.ROI_MASK_HEAD.RESOLUTION
            mask_pooler_resolution        = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
            mask_pooler_scales            = pooler_scales
            mask_sampling_ratio           = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        for c in in_channels:
            assert c == in_channels[0]
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution, scales=pooler_scales, sampling_ratio=sampling_ratio
        )
        self.box_head = build_box_head(cfg, (in_channels, pooler_resolution, pooler_resolution))

        self.box_predictor = FastRCNNOutputHead(self.box_head.output_size, num_classes)
        self.box2box_transform = Box2BoxTransform(weights=bbox_reg_weights)

        if self.mask_on:
            self.mask_pooler = ROIPooler(
                output_size=mask_pooler_resolution,
                scales=mask_pooler_scales,
                sampling_ratio=mask_sampling_ratio,
            )
            self.mask_head = build_mask_head(cfg, in_channels)

    def forward(self, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        features = [features[f] for f in self.in_features]
        if self.training:
            proposals, targets = self.sample_proposals_for_training(proposals, targets)

        box_features = self.box_pooler(features, proposals)
        box_features = self.box_head(box_features)
        class_logits, regression_outputs = self.box_predictor(box_features)
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform, class_logits, regression_outputs, proposals, targets
        )

        if self.training:
            losses = outputs.losses()
            if self.mask_on:
                # During training the same proposals used by the box head are used by
                # the mask head. The loss is only defined on positive proposals.
                proposals, targets, _ = keep_only_positive_boxes(proposals, targets)
                mask_features = self.mask_pooler(features, proposals)
                mask_logits = self.mask_head(mask_features)
                losses["loss_mask"] = maskrcnn_loss(
                    proposals, mask_logits, targets, self.mask_discretization_size
                )
            return proposals, losses
        else:
            losses = {}
            detections = fastrcnn_inference(
                outputs.predict_boxes(),
                outputs.predict_probs(),
                outputs.image_shapes,
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            if self.mask_on:
                # During inference cascaded prediction is used: the mask head is only
                # applied to the top scoring box detections.
                mask_features = self.mask_pooler(features, detections)
                mask_logits = self.mask_head(mask_features)
                maskrcnn_inference(mask_logits, detections)
            return detections, losses
