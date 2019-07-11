import math
from collections import defaultdict
import torch
from borc.nn.focal_loss import sigmoid_focal_loss_jit
from torch import nn

from detectron2.layers import cat, nms, smooth_l1_loss
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou

from ..anchor_generator import build_anchor_generator
from ..backbone import build_backbone
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..postprocessing import detector_postprocess
from .model_builder import META_ARCH_REGISTRY

__all__ = ["RetinaNet"]


def _permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)  # Size=(N,HWA,C)
    return layer


def _concat_box_prediction_layers(box_cls, box_regression):
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = _permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = _permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, C)
    box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


@META_ARCH_REGISTRY.register()
class RetinaNet(nn.Module):
    """
    RetinaNet head. Creates FPN backbone, anchors and a head for classification
    and box regression. Calculates and applies smooth L1 loss and sigmoid focal
    loss.
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.head = RetinaNetHead(cfg)
        self.anchor_generator = build_anchor_generator(cfg)

        # fmt: off
        self.num_classes              = cfg.MODEL.RETINANET.NUM_CLASSES
        self.in_features              = cfg.MODEL.RETINANET.IN_FEATURES
        # Loss parameters:
        self.focal_loss_alpha         = cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma         = cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta      = cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA
        # Inference parameters:
        self.score_threshold          = cfg.MODEL.RETINANET.INFERENCE_SCORE_THRESHOLD
        self.topk_candidates          = cfg.MODEL.RETINANET.INFERENCE_TOPK_CANDIDATES
        self.nms_threshold            = cfg.MODEL.RETINANET.INFERENCE_NMS_THRESHOLD
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMG
        # fmt: on

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.matcher = Matcher(
            cfg.MODEL.RETINANET.FG_IOU_THRESHOLD,
            cfg.MODEL.RETINANET.BG_IOU_THRESHOLD,
            allow_low_quality_matches=True,
        )

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DetectionTransform` .
                Each item in the list contains the inputs for one image.
            For now, each item in the list is a dict that contains:
                image: Tensor, image in (C, H, W) format.
                targets: Instances
                Other information that's included in the original dicts, such as:
                    "height", "width" (int): the output resolution of the model, used in inference.
                        See :meth:`postprocess` for details.
         Returns:
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if "targets" in batched_inputs[0]:
            targets = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            targets = None

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        box_cls, box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)

        if self.training:
            gt_classes, gt_anchors_reg_deltas = self.get_ground_truth(anchors, targets)
            return self.losses(gt_classes, gt_anchors_reg_deltas, box_cls, box_regression)
        else:
            return self.inference(box_cls, box_regression, anchors, images, batched_inputs)

    def losses(self, gt_classes, gt_anchors_deltas, pred_class_logits, pred_anchor_deltas):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            For `pred_class_logits` and `pred_anchor_deltas`, see
                :meth:`RetinaNetHead.forward`.

        Returns:
            losses (dict[str: Tensor]): mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_class_logits, pred_anchor_deltas = _concat_box_prediction_layers(
            pred_class_logits, pred_anchor_deltas
        )
        gt_classes = cat(gt_classes)
        gt_anchors_deltas = cat(gt_anchors_deltas)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # logits loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1, num_foreground)

        # regression loss
        loss_box_reg = smooth_l1_loss(
            pred_anchor_deltas[foreground_idxs],
            gt_anchors_deltas[foreground_idxs],
            beta=self.smooth_l1_loss_beta,
        ) / max(1, num_foreground)

        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}

    def get_ground_truth(self, anchors, targets):
        """
        Args:
            anchors (list[list[Boxes]]): a list of #image elements. Each is a
                list of #feature level Boxes. The Boxes contains anchors of
                this image on the specific feature level.
            targets (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.

        Returns:
            gt_classes (list[Tensor]): A tensor of shape (R,) storing ground-truth classification
                labels. R is the number of anchors. Anchors that are assigned one of the
                C labels with a intersection-over-union (IoU) i.e. a metric of confidence that is
                higher than the foreground threshold are assigned their corresponding label in the
                [0, C-1] range. Anchors whose IoU are below the background threshold are assigned
                the label "C". Anchors whose IoU are between the foreground and background
                thresholds are assigned a label "-1".
            gt_anchors_deltas (list[Tensor]): shape (R, 4), row i represents ground-truth box2box
                transform targets (dx, dy, dw, dh) that map object instance i to its matched
                ground-truth box.
        """
        gt_classes = []
        gt_anchors_deltas = []
        anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]  # list of boxes for each image
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, anchors_per_image)
            gt_matched_idxs = self.matcher(match_quality_matrix)

            # ground truth box regression
            matched_gt_boxes = targets_per_image[gt_matched_idxs.clamp(min=0)].gt_boxes
            gt_anchors_reg_deltas_i = self.box2box_transform.get_deltas(
                anchors_per_image.tensor, matched_gt_boxes.tensor
            )

            # ground truth classes
            has_gt = len(targets_per_image) > 0
            if has_gt:
                # Clamping assumes at least one ground truth, not necessarily true.
                matched_idxs_clamped = gt_matched_idxs.clamp(min=0)
                gt_classes_i = targets_per_image.gt_classes[matched_idxs_clamped]
                # Anchors that are below the threshold are treated as background.
                gt_classes_i[gt_matched_idxs == Matcher.BELOW_LOW_THRESHOLD] = self.num_classes
                # Anchors that are between thresholds are ignored during training.
                gt_classes_i[gt_matched_idxs == Matcher.BETWEEN_THRESHOLDS] = -1
            else:
                gt_classes_i = torch.zeros_like(gt_matched_idxs) + self.num_classes

            gt_classes.append(gt_classes_i)
            gt_anchors_deltas.append(gt_anchors_reg_deltas_i)

        return gt_classes, gt_anchors_deltas

    def inference(
        self, box_cls, box_regression, anchors, images, batched_inputs, do_postprocess=True
    ):
        """
        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                a tensor of size N, A * C, H, W. Here, N is the number of images,
                A is the number of anchors at every pixel per level (= # of
                aspect ratios * # of anchor sizes), C is the number of classes,
                H is the image height and W is the image width.
            box_regression (list[Tensor]): Same as `box_cls` parameter above.
                C is 4 since it contains dx, dy, dw, dh values.
            anchors (list[list[Boxes]]): a list of #images elements. Each is a
                list of #feature level Boxes. The Boxes contain anchors of this
                image on the specific feature level.
            images (Tensor): image in (C, H, W) format.
            batched_inputs (list[dict]): same as in :meth:`forward`
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        predicted_boxes = []
        for img_idx, anchors_per_image in enumerate(anchors):
            image_size = images.image_sizes[img_idx]
            box_cls_per_image = [box_cls_per_layer[img_idx] for box_cls_per_layer in box_cls]
            box_reg_per_image = [box_reg_per_layer[img_idx] for box_reg_per_layer in box_regression]
            predicted_boxes_per_image = self.inference_single_image(
                box_cls_per_image, box_reg_per_image, anchors_per_image, tuple(image_size)
            )

            # post-process the output boxes per image
            if do_postprocess:
                height = batched_inputs[img_idx].get("height", image_size[0])
                width = batched_inputs[img_idx].get("width", image_size[1])
                postprocessed_predicted_boxes_per_image = detector_postprocess(
                    predicted_boxes_per_image, height, width
                )
                predicted_boxes.append({"instances": postprocessed_predicted_boxes_per_image})
            else:
                predicted_boxes.append({"instances": predicted_boxes_per_image})
        return predicted_boxes

    def inference_single_image(self, box_cls, box_regression, anchors, image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size A * C, H, W.
            box_regression (list[Tensor]): Same as 'box_cls' parameter. C is 4.
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = defaultdict(list)  # will contain (tensor([x,y,w,h]), tensor([score]))

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_regression, anchors):
            _, H, W = box_cls_i.shape
            A = box_reg_i.size(0) // 4
            C = box_cls_i.size(0) // A

            box_cls_i = _permute_and_flatten(box_cls_i, 1, A, C, H, W).flatten()
            box_reg_i = _permute_and_flatten(box_reg_i, 1, A, 4, H, W).reshape(-1, 4)

            # predict scores
            predicted_prob = box_cls_i.sigmoid()
            zeros = torch.zeros(predicted_prob.shape).cuda()
            predicted_prob = torch.where(
                predicted_prob > self.score_threshold, predicted_prob, zeros
            )

            # Keep top k top scoring indices only.
            num_anchors = A * H * W
            k = min(self.topk_candidates, num_anchors)
            predicted_prob, topk_idxs = predicted_prob.topk(k, sorted=True, dim=0)

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i.tensor)

            # iterate over topk anchor (with unique class) that passed thresholding
            for anchor_class, box, score in zip(classes_idxs, predicted_boxes, predicted_prob):
                device = box.device if isinstance(score, torch.Tensor) else torch.device("cpu")
                score = torch.tensor([score], device=device)
                anchor_class = int(anchor_class)
                if anchor_class in boxes_all.keys():
                    boxes_all[anchor_class].append((box, score))
                else:
                    boxes_all[anchor_class] = [(box, score)]

        # Combine predictions across all levels and retain the top scoring by class
        results_per_anchor = []
        for anchor_classes, boxes_and_scores in boxes_all.items():
            # All Instance values need to be a Tensor or a `Box` object.
            boxes, scores = zip(*boxes_and_scores)
            boxes = torch.cat(boxes, 0).reshape(-1, 4)
            scores = torch.cat(scores)
            anchor_classes = torch.tensor(
                [anchor_classes for _ in range(scores.shape[0])], device=scores.device
            )

            keep_idxs = nms(boxes, scores, self.nms_threshold)

            result = Instances(image_size)
            result.pred_boxes = Boxes(boxes[keep_idxs])
            result.scores = scores[keep_idxs]
            result.pred_classes = anchor_classes[keep_idxs]
            results_per_anchor.append(result)

        results = Instances.cat(results_per_anchor)

        # Limit to max_detections_per_image detections **over all object classes**
        num_detections = len(results)
        if num_detections > self.max_detections_per_image > 0:
            cls_scores = results.scores
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), num_detections - self.max_detections_per_image + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            results = results[keep]
        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class RetinaNetHead(nn.Module):
    def __init__(self, cfg):
        """
        Creates a head for object classification subnet and box regression
        subnet. Although the two subnets share a common structure, they use
        separate parameters (unlike RPN).
        """
        super().__init__()
        # fmt: off
        self.in_features = cfg.MODEL.RETINANET.IN_FEATURES
        in_channels      = cfg.MODEL.FPN.OUT_CHANNELS
        num_classes      = cfg.MODEL.RETINANET.NUM_CLASSES
        num_convs        = cfg.MODEL.RETINANET.NUM_CONVS
        prior_prob       = cfg.MODEL.RETINANET.PRIOR_PROB
        num_anchors      = len(cfg.MODEL.RETINANET.ANCHOR_ASPECT_RATIOS[0]) * \
            cfg.MODEL.RETINANET.SCALES_PER_OCTAVE
        # fmt: on
        # aspect ratio list ANCHOR_ASPECT_RATIOS[0] is used for all IN_FEATURES.
        assert (
            len(cfg.MODEL.RETINANET.ANCHOR_ASPECT_RATIOS) == 1
        ), "Using different aspect ratios between levels is not currently supported!"

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): predicted the probability of object presence
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): predicted 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        logits = []
        bbox_reg = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg
