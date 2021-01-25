# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List
import torch
from fvcore.nn import sigmoid_focal_loss_jit
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.box_regression import Box2BoxTransform, _dense_box_regression_loss
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.retinanet import RetinaNetHead, permute_to_N_HWA_K
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.comm import get_world_size
from detectron2.utils.events import get_event_storage

__all__ = ["ATSS"]

INF = 100000000


def reduce_sum(tensor):
    if get_world_size() <= 1:
        return tensor
    import torch.distributed as dist

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


@META_ARCH_REGISTRY.register()
class ATSS(nn.Module):
    """
    Implement ATSS in :paper:`ATSS`.
    """

    def __init__(self, cfg):
        super().__init__()
        # fmt: off
        self.num_classes              = cfg.MODEL.ATSS.NUM_CLASSES
        self.in_features              = cfg.MODEL.ATSS.IN_FEATURES
        # Loss parameters:
        self.focal_loss_alpha         = cfg.MODEL.ATSS.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma         = cfg.MODEL.ATSS.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta      = cfg.MODEL.ATSS.SMOOTH_L1_LOSS_BETA
        self.box_reg_loss_type        = cfg.MODEL.ATSS.BBOX_REG_LOSS_TYPE
        self.reg_loss_weight          = cfg.MODEL.ATSS.REG_LOSS_WEIGHT
        self.selection_mode           = cfg.MODEL.ATSS.SELECTION_MODE
        self.topk                     = cfg.MODEL.ATSS.TOPK
        # Inference parameters:
        self.score_threshold          = cfg.MODEL.ATSS.SCORE_THRESH_TEST
        self.topk_candidates          = cfg.MODEL.ATSS.TOPK_CANDIDATES_TEST
        self.nms_threshold            = cfg.MODEL.ATSS.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # fmt: on

        self.backbone = build_backbone(cfg)

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = ATSSHead(cfg, feature_shapes)
        self.anchor_generator = build_anchor_generator(cfg, feature_shapes)

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(
            cfg.MODEL.ATSS.IOU_THRESHOLDS, cfg.MODEL.ATSS.IOU_LABELS, allow_low_quality_matches=True
        )

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]

        anchors = self.anchor_generator(features)
        pred_logits, pred_anchor_deltas, centerness = self.head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]
        centerness = [permute_to_N_HWA_K(x, 1) for x in centerness]

        if self.training:
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)
            losses = self.losses(
                anchors, pred_logits, gt_labels, pred_anchor_deltas, centerness, gt_boxes
            )

            return losses
        else:
            results = self.inference(
                anchors, pred_logits, pred_anchor_deltas, centerness, images.image_sizes
            )
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def compute_centerness_targets(self, gt_boxes, anchors):
        """
        Args:
            gt_boxes, anchors: Tensors of size (X, 4)

        Returns:
            Tensor: Size X of centerness targets
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        # If the center of the anchor is outside the gt box
        # we get a negative value, min with 0 to avoid nan
        l = torch.clamp(anchors_cx - gt_boxes[:, 0], min=0)
        t = torch.clamp(anchors_cy - gt_boxes[:, 1], min=0)
        r = torch.clamp(gt_boxes[:, 2] - anchors_cx, min=0)
        b = torch.clamp(gt_boxes[:, 3] - anchors_cy, min=0)
        left_right = torch.stack([l, r], dim=1)
        top_bottom = torch.stack([t, b], dim=1)
        centerness = torch.sqrt(
            (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0])
            * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        )
        assert not torch.isnan(centerness).any()
        return centerness

    def losses(self, anchors, pred_logits, gt_labels, pred_anchor_deltas, centerness, gt_boxes):
        """
        Args:
            anchors (list[Boxes]): a list of #feature level Boxes
            gt_labels, gt_boxes: see output of :meth:`ATSS.label_anchors`.
                Their shapes are (N, R) and (N, R, 4), respectively, where R is
                the total number of anchors across levels, i.e. sum(Hi x Wi x Ai)
            pred_logits, pred_anchor_deltas, centerness: are list[Tensor]. Each element in the
                list corresponds to one level and has shape (N, Hi * Wi * Ai, K or 4 or 1).
                Where K is the number of classes used in `pred_logits`.

        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg" and "loss_centerness"
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, R)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)

        # classification and regression loss
        gt_labels_target = F.one_hot(gt_labels[valid_mask], num_classes=self.num_classes + 1)[
            :, :-1
        ]  # no loss for the last (background) class
        loss_cls = sigmoid_focal_loss_jit(
            cat(pred_logits, dim=1)[valid_mask],
            gt_labels_target.to(pred_logits[0].dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )

        loss_box_reg = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_loss_beta,
        )

        anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)
        collapsed_pos_mask = pos_mask.view(-1)
        collapsed_gt_boxes = cat(gt_boxes)[collapsed_pos_mask]

        # Compute num_pos_avg_per_gpu to adjust loss_cls
        num_gpus = get_world_size()
        valid_indices = torch.nonzero(collapsed_pos_mask).squeeze(1)
        total_num_pos = reduce_sum(valid_indices.new_tensor([valid_indices.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        # Upscale anchors from Rx4 to (NxR)x4
        collapsed_anchors = cat([anchors for _ in range(num_images)])[collapsed_pos_mask]
        centerness_targets = self.compute_centerness_targets(collapsed_gt_boxes, collapsed_anchors)
        sum_centerness_targets_avg_per_gpu = reduce_sum(centerness_targets.sum()).item() / float(
            num_gpus
        )

        if centerness_targets.sum() > 0:
            loss_box_reg = (
                loss_box_reg * centerness_targets
            ).sum() / sum_centerness_targets_avg_per_gpu
        else:
            loss_box_reg = loss_box_reg.sum()

        collapsed_centerness = cat(centerness, dim=1).squeeze(-1).view(-1)[collapsed_pos_mask]

        loss_cls = loss_cls / num_pos_avg_per_gpu
        loss_centerness = (
            F.binary_cross_entropy_with_logits(
                collapsed_centerness, centerness_targets, reduction="sum"
            )
            / num_pos_avg_per_gpu
        )

        return {
            "loss_cls": loss_cls,
            "loss_box_reg": self.reg_loss_weight * loss_box_reg,
            "loss_centerness": loss_centerness,
        }

    @torch.no_grad()
    def label_anchors(self, anchors, gt_instances):
        """
        Args:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
            gt_instances (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps (sum(Hi * Wi * A)).
                Label values are in {-1, 0, ..., K}, with -1 means ignore, and K means background.
            list[Tensor]:
                i-th element is a Rx4 tensor, where R is the total number of anchors across
                feature maps. The values are the matched gt boxes for each anchor.
                Values are undefined for those anchors not labeled as foreground.
        """
        leveled_anchors = anchors
        anchors = Boxes.cat(anchors)  # Rx4

        gt_labels = []
        matched_gt_boxes = []
        for gt_per_image in gt_instances:

            if len(gt_per_image) > 0:
                gt_tensor = gt_per_image.gt_boxes.tensor

                gt_cx = (gt_tensor[:, 2] + gt_tensor[:, 0]) / 2.0
                gt_cy = (gt_tensor[:, 3] + gt_tensor[:, 1]) / 2.0
                gt_points = torch.stack((gt_cx, gt_cy), dim=1)

                anchors_cx_per_im = (anchors.tensor[:, 2] + anchors.tensor[:, 0]) / 2.0
                anchors_cy_per_im = (anchors.tensor[:, 3] + anchors.tensor[:, 1]) / 2.0
                anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)

                distances = (
                    (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()
                )

                num_gt = gt_tensor.shape[0]
                ious = pairwise_iou(anchors, gt_per_image.gt_boxes)

                if self.selection_mode == "iou":
                    topk = min(self.topk, ious.size(dim=0))
                    _, candidate_idxs = ious.topk(topk, dim=0, largest=True)
                    # Probably a better way to do this, >= always true
                    candidate_ious = ious[candidate_idxs, torch.arange(num_gt)]
                    is_pos = candidate_ious >= 0
                if self.selection_mode == "distance":
                    topk = min(self.topk, distances.size(dim=0))
                    _, candidate_idxs = distances.topk(topk, dim=0, largest=False)
                    # Probably a better way to do this, >= always true
                    candidate_ious = ious[candidate_idxs, torch.arange(num_gt)]
                    is_pos = candidate_ious >= 0
                else:
                    # Selecting candidates based on the center distance between anchor and object
                    num_anchors_per_level = [
                        len(anchors_per_level.tensor) for anchors_per_level in leveled_anchors
                    ]
                    candidate_idxs = []
                    start_idx = 0
                    for level, _ in enumerate(leveled_anchors):
                        end_idx = start_idx + num_anchors_per_level[level]
                        distances_per_level = distances[start_idx:end_idx, :]
                        topk = min(self.topk, num_anchors_per_level[level])
                        _, topk_idxs_per_level = distances_per_level.topk(
                            topk, dim=0, largest=False
                        )
                        candidate_idxs.append(topk_idxs_per_level + start_idx)
                        start_idx = end_idx
                    candidate_idxs = torch.cat(candidate_idxs, dim=0)

                    # Using the sum of mean and standard deviation
                    # as the IoU threshold to select final positive samples
                    candidate_ious = ious[candidate_idxs, torch.arange(num_gt)]
                    iou_mean_per_gt = candidate_ious.mean(0)
                    iou_std_per_gt = candidate_ious.std(0)
                    iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
                    is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

                # Limiting the final positive samples’ center to object
                anchor_num = anchors_cx_per_im.shape[0]
                for ng in range(num_gt):
                    candidate_idxs[:, ng] += ng * anchor_num
                e_anchors_cx = (
                    anchors_cx_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
                )
                e_anchors_cy = (
                    anchors_cy_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
                )
                candidate_idxs = candidate_idxs.view(-1)
                l = e_anchors_cx[candidate_idxs].view(-1, num_gt) - gt_tensor[:, 0]
                t = e_anchors_cy[candidate_idxs].view(-1, num_gt) - gt_tensor[:, 1]
                r = gt_tensor[:, 2] - e_anchors_cx[candidate_idxs].view(-1, num_gt)
                b = gt_tensor[:, 3] - e_anchors_cy[candidate_idxs].view(-1, num_gt)
                is_in_gts = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
                is_pos = is_pos & is_in_gts

                # if an anchor box is assigned to multiple gts,
                # the one with the highest IoU will be selected.
                ious_inf = torch.full_like(ious, -INF).t().contiguous().view(-1)
                index = candidate_idxs.view(-1)[is_pos.view(-1)]
                ious_inf[index] = ious.t().contiguous().view(-1)[index]
                ious_inf = ious_inf.view(num_gt, -1).t()

                anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)
                gt_labels_i = gt_per_image.gt_classes[anchors_to_gt_indexs]
                gt_labels_i[anchors_to_gt_values == -INF] = self.num_classes
                matched_gt_boxes_i = gt_tensor[anchors_to_gt_indexs]
            else:
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
                gt_labels_i = torch.zeros_like(anchors.tensor[:, 0]) + self.num_classes

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes

    def inference(self, anchors, pred_logits, pred_anchor_deltas, centerness, image_sizes):
        """
        Arguments:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contain anchors of this image on the specific feature level.
            pred_logits, pred_anchor_deltas, centerness: list[Tensor], one per level. Each
                has shape (N, Hi * Wi * Ai, K or 4 or 1)
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results = []
        for img_idx, image_size in enumerate(image_sizes):
            pred_logits_per_image = [x[img_idx] for x in pred_logits]
            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
            centerness_per_image = [x[img_idx] for x in centerness]
            results_per_image = self.inference_single_image(
                anchors,
                pred_logits_per_image,
                deltas_per_image,
                centerness_per_image,
                tuple(image_size),
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, anchors, box_cls, box_delta, centerness, image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors in that feature level.
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            centerness (list[Tensor]): Same shape as 'box_cls' except that K becomes 1.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, centerness_i, anchors_i in zip(
            box_cls, box_delta, centerness, anchors
        ):
            # (HxWxAxK,)
            box_cls_i = (box_cls_i.sigmoid_() * centerness_i.sigmoid_()).flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class ATSSHead(RetinaNetHead):
    """
    The head used in ATSS for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    @configurable
    def __init__(
        self, *, input_shape: List[ShapeSpec], conv_dims: List[int], num_anchors, **kwargs
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (List[ShapeSpec]): input shape
            conv_dims (List[int]): dimensions for each convolution layer
            num_anchors (int): number of generated anchors
            kwargs: Remaining input args for RetinaNetHead init
        """
        super().__init__(
            input_shape=input_shape, conv_dims=conv_dims, num_anchors=num_anchors, **kwargs
        )

        self.centerness = nn.Conv2d(
            conv_dims[-1], num_anchors * 1, kernel_size=3, stride=1, padding=1
        )

        # Initialization
        torch.nn.init.normal_(self.centerness.weight, mean=0, std=0.01)
        torch.nn.init.constant_(self.centerness.bias, 0)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(len(input_shape))])

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        num_anchors = build_anchor_generator(cfg, input_shape).num_cell_anchors
        # fmt: on
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        return {
            "input_shape": input_shape,
            "num_classes": cfg.MODEL.ATSS.NUM_CLASSES,
            "conv_dims": [input_shape[0].channels] * cfg.MODEL.ATSS.NUM_CONVS,
            "prior_prob": cfg.MODEL.ATSS.PRIOR_PROB,
            "norm": cfg.MODEL.ATSS.NORM,
            "num_anchors": num_anchors,
        }

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
            centerness (list[Tensor]): #lvl tensors, each has shape (N, Ax1, Hi, Wi).
                The tensor predicts the centerness for every anchor.
        """
        logits = []
        bbox_reg = []
        centerness = []
        for i, feature in enumerate(features):
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_subnet = self.bbox_subnet(feature)
            bbox_reg.append(self.scales[i](self.bbox_pred(bbox_subnet)))
            centerness.append(self.centerness(bbox_subnet))
        return logits, bbox_reg, centerness
