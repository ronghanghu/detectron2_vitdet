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
from .fast_rcnn import FastRCNNOutputHead, FastRCNNOutputs, fast_rcnn_inference
from .mask_head import build_mask_head, mask_rcnn_inference, mask_rcnn_loss


ROI_HEADS_REGISTRY = Registry("ROI_HEADS")


def build_roi_heads(cfg):
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg)


def select_foreground_proposals(box_lists):
    """
    Given a list of N BoxLists (for N images), each containing a `classes_gt` field,
    return a list of BoxLists that contain only boxes with `classes_gt > 0`.

    Args:
        box_lists (list[BoxList]): A list of N BoxLists, where N is the number of
            images in the batch.
    """
    assert isinstance(box_lists, (list, tuple))
    assert isinstance(box_lists[0], BoxList)
    assert box_lists[0].has_field("classes_gt")
    fg_box_lists = []
    fg_selection_masks = []
    for box_list_per_image in box_lists:
        classes_gt = box_list_per_image.get_field("classes_gt")
        fg_selection_mask = classes_gt > 0
        fg_inds = fg_selection_mask.nonzero().squeeze(1)
        fg_box_lists.append(box_list_per_image[fg_inds])
        fg_selection_masks.append(fg_selection_mask)
    return fg_box_lists, fg_selection_masks


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

    def label_and_sample_proposals(self, proposals, targets):
        """
        Perform box matching between `proposals` and `targets`, and label the
        proposals with training labels. Return `self.batch_size_per_image`
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
                sampled for training. Each `BoxList` has the following fields:
                - classes_gt: labeling each box with a category ranging in [0, #class].
                - boxes_gt: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                - masks_gt: the ground-truth mask of the box that the proposal is
                  assigned to.
        """
        proposals_with_gt = []

        with torch.no_grad():
            for proposals_per_image, targets_per_image in zip(proposals, targets):
                match_quality_matrix = boxlist_iou(targets_per_image, proposals_per_image)
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # Fast RCNN only need "classes_gt" field for selecting the targets
                # Get the targets corresponding GT for each proposal
                # Need to clamp the indices because matched_idxs can be < 0
                matched_idxs_clamped = matched_idxs.clamp(min=0)

                classes_gt = targets_per_image.get_field("classes_gt")[matched_idxs_clamped].to(
                    dtype=torch.int64
                )

                # Label background (below the low threshold)
                classes_gt[matched_idxs == Matcher.BELOW_LOW_THRESHOLD] = 0
                # Label ignore proposals (between low and high thresholds)
                classes_gt[matched_idxs == Matcher.BETWEEN_THRESHOLDS] = -1

                sampled_fg_inds, sampled_bg_inds = subsample_labels(
                    classes_gt, self.batch_size_per_image, self.positive_sample_fraction
                )
                sampled_inds = torch.cat([sampled_fg_inds, sampled_bg_inds], dim=0)

                proposals_per_image = proposals_per_image[sampled_inds]
                proposals_per_image.add_field("classes_gt", classes_gt[sampled_inds])

                # Avoid indexing the BoxList targets_per_image directly, as in the
                # commented-out row below. It is considerably slower.
                # TODO: investigate and optimize this access pattern.
                # boxes_gt = targets_per_image[matched_idxs_clamped[sampled_inds]].bbox  # slow!
                boxes_gt = targets_per_image.bbox[matched_idxs_clamped[sampled_inds]]
                proposals_per_image.add_field("boxes_gt", boxes_gt)

                if targets_per_image.has_field("masks_gt"):
                    # See note above about not indexing the targets_per_image directly
                    masks_gt = targets_per_image.get_field("masks_gt")[
                        matched_idxs_clamped[sampled_inds]
                    ]
                    proposals_per_image.add_field("masks_gt", masks_gt)

                proposals_with_gt.append(proposals_per_image)

        return proposals_with_gt

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
                detected objects. Returned during inference only; may be []
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
            proposals = self.label_and_sample_proposals(proposals, targets)

        box_features = self._shared_roi_transform(features, proposals)
        feature_pooled = F.avg_pool2d(box_features, 7)  # gap
        class_logits_pred, box_deltas_pred = self.box_predictor(feature_pooled)
        del feature_pooled

        outputs = FastRCNNOutputs(
            self.box2box_transform, class_logits_pred, box_deltas_pred, proposals
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
            box_lists_pred = fast_rcnn_inference(
                outputs.predict_boxes(),
                outputs.predict_probs(),
                outputs.image_shapes,
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            if self.mask_on:
                x = self._shared_roi_transform(features, box_lists_pred)
                mask_logits = self.mask_head(x)
                mask_rcnn_inference(mask_logits, box_lists_pred)
            return box_lists_pred, {}


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
        self.mask_side_len       = cfg.MODEL.ROI_MASK_HEAD.RESOLUTION
        mask_pooler_resolution   = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        mask_pooler_scales       = pooler_scales
        mask_sampling_ratio      = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
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
            proposals = self.label_and_sample_proposals(proposals, targets)

        box_features = self.box_pooler(features, proposals)
        box_features = self.box_head(box_features)
        class_logits_pred, box_deltas_pred = self.box_predictor(box_features)
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform, class_logits_pred, box_deltas_pred, proposals
        )

        if self.training:
            losses = outputs.losses()
            if self.mask_on:
                # During training the same proposals used by the box head are used by
                # the mask head. The loss is only defined on positive proposals.
                proposals, _ = select_foreground_proposals(proposals)
                mask_features = self.mask_pooler(features, proposals)
                mask_logits = self.mask_head(mask_features)
                losses["loss_mask"] = mask_rcnn_loss(mask_logits, proposals, self.mask_side_len)
            return [], losses
        else:
            box_lists_pred = fast_rcnn_inference(
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
                mask_features = self.mask_pooler(features, box_lists_pred)
                mask_logits = self.mask_head(mask_features)
                mask_rcnn_inference(mask_logits, box_lists_pred)
            return box_lists_pred, {}
