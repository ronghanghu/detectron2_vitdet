# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch

from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou

from .track_head import build_track_head

logger = logging.getLogger(__name__)


@ROI_HEADS_REGISTRY.register()
class MaskTrackROIHeads(StandardROIHeads):
    """
    ROI heads for MaskTrackRCNN model. Includes standard box and mask
    heads from MaskRCNN as well as custom tracking head and memory queue.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self._init_mask_track(cfg, input_shape)

    def _init_mask_track(self, cfg, input_shape):
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        # fmt: on

        # If MaskTrackROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]
        self.track_head = build_track_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        # memory queue for testing
        self.prev_bboxes = None
        self.prev_roi_feats = None

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        batched_inputs: List[Dict] = None,
        targets: Optional[List[Instances]] = None,
        ref_features: Optional[Dict[str, torch.Tensor]] = None,
        ref_targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward` and :class:`MaskTrackROIHeads._forward_box`.
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals, None, ref_features, ref_targets)
            losses.update(self._forward_mask(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals, batched_inputs)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        batched_inputs: List[Dict] = None,
        ref_features: Optional[Dict[str, torch.Tensor]] = None,
        ref_targets: Optional[List[Instances]] = None,
    ):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
            batched_inputs (list[Dict]): batched list of inputs for model
            ref_features: (Optional[dict[str, Tensor]]): only used at training time,
                mapping from feature map name to feature of reference images.
                Similar to features in :class:`MaskTrackROIHeads.forward`.
            ref_targets: (Optional[list[Instances]]): only used at training time,
                list of `Instances` which contain ground-truth per-instance annotations
                for reference images. Similar to targets in :class:`MaskTrackROIHeads.forward`.

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """

        features = [features[f] for f in self.box_in_features]
        box_features_pooled = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features_pooled)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)

            # track head
            ref_features = [ref_features[f] for f in self.box_in_features]
            ref_box_features_pooled = self.box_pooler(
                ref_features, [x.gt_boxes for x in ref_targets]
            )

            bbox_img_n = [len(p) for p in proposals]
            ref_bbox_img_n = [len(t) for t in ref_targets]

            # only considers foreground objects when computing loss_match
            pid_weights = [(p.gt_classes != self.num_classes).double() for p in proposals]
            pids = torch.cat([p.gt_pids * w.int() for (p, w) in zip(proposals, pid_weights)])
            match_score = self.track_head(
                box_features_pooled, ref_box_features_pooled, bbox_img_n, ref_bbox_img_n
            )
            loss_match = self.track_head.loss(match_score, pids, pid_weights)
            losses.update(loss_match)
            del box_features_pooled
            del ref_features
            del ref_box_features_pooled

            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            del box_features_pooled
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.track_instances(pred_instances, batched_inputs, features)
            return pred_instances

    def track_instances(self, pred_instances, batched_inputs, features):
        assert len(pred_instances) == 1
        assert len(batched_inputs) == 1
        is_first = batched_inputs[0]["is_first"]

        pred_boxes = pred_instances[0].pred_boxes
        det_labels = pred_instances[0].pred_classes

        det_bboxes = pred_boxes.tensor
        det_roi_feats = self.box_pooler(features, [pred_boxes])

        if len(pred_boxes) == 0:
            det_obj_ids = torch.tensor([], dtype=torch.int64)
            if is_first:
                self.prev_bboxes = None
                self.prev_roi_feats = None
                self.prev_det_labels = None

        elif is_first or (not is_first and self.prev_bboxes is None):
            det_obj_ids = torch.arange(det_bboxes.size(0))
            # save bbox and features for later matching
            self.prev_bboxes = det_bboxes
            self.prev_roi_feats = det_roi_feats
            self.prev_det_labels = det_labels
        else:
            assert self.prev_roi_feats is not None
            # only support one image at a time
            bbox_img_n = [det_bboxes.size(0)]
            prev_bbox_img_n = [self.prev_roi_feats.size(0)]
            match_score = self.track_head(
                det_roi_feats, self.prev_roi_feats, bbox_img_n, prev_bbox_img_n
            )[0]
            match_logprob = torch.nn.functional.log_softmax(match_score, dim=1)
            label_delta = (self.prev_det_labels == det_labels.view(-1, 1)).float()
            bbox_ious = pairwise_iou(Boxes(det_bboxes[:, :4]), Boxes(self.prev_bboxes[:, :4]))
            # compute comprehensive score
            comp_scores = self.track_head.compute_comp_scores(
                match_logprob,
                pred_instances[0].scores.view(-1, 1),
                bbox_ious,
                label_delta,
                add_bbox_dummy=True,
            )
            match_likelihood, match_ids = torch.max(comp_scores, dim=1)
            # translate match_ids to det_obj_ids, assign new id to new objects
            # update tracking features/bboxes of exisiting object,
            # add tracking features/bboxes of new object
            match_ids = match_ids.cpu().numpy().astype(np.int32)
            det_obj_ids = torch.ones((match_ids.shape[0]), dtype=torch.int32) * (-1)
            best_match_scores = np.ones((self.prev_bboxes.size(0))) * (-100)
            for idx, match_id in enumerate(match_ids):
                if match_id == 0:
                    # add new object
                    det_obj_ids[idx] = self.prev_roi_feats.size(0)
                    self.prev_roi_feats = torch.cat(
                        (self.prev_roi_feats, det_roi_feats[idx][None]), dim=0
                    )
                    self.prev_bboxes = torch.cat((self.prev_bboxes, det_bboxes[idx][None]), dim=0)
                    self.prev_det_labels = torch.cat(
                        (self.prev_det_labels, det_labels[idx][None]), dim=0
                    )
                else:
                    # multiple candidates might match with previous object,
                    # here we choose the one with the largest comprehensive score
                    obj_id = match_id - 1
                    match_score = comp_scores[idx, match_id]
                    if match_score > best_match_scores[obj_id]:
                        det_obj_ids[idx] = obj_id
                        best_match_scores[obj_id] = match_score
                        # update feature
                        self.prev_roi_feats[obj_id] = det_roi_feats[idx]
                        self.prev_bboxes[obj_id] = det_bboxes[idx]

        pred_instances[0].pred_obj_ids = det_obj_ids.to(pred_boxes.device)
        return pred_instances
