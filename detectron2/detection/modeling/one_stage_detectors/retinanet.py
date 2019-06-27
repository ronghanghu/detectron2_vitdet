import torch
from torch import nn
import math

from borc.nn.focal_loss import sigmoid_focal_loss_jit
from detectron2.structures import Boxes, pairwise_iou
from detectron2.detection.modeling.backbone import build_backbone
from detectron2.layers import cat, smooth_l1_loss
from detectron2.structures import ImageList
from ..anchor_generator import build_anchor_generator
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..model_builder import META_ARCH_REGISTRY

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
    for box_cls_per_level, box_regression_per_level in zip(
        box_cls, box_regression
    ):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = _permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = _permute_and_flatten(
            box_regression_per_level, N, A, 4, H, W
        )
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
        super(RetinaNet, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.box_cls_and_regression = RetinaNetClassifierRegressionHead(cfg)
        self.anchor_generator = build_anchor_generator(cfg)

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.matcher = Matcher(
            cfg.MODEL.RETINANET.FG_IOU_THRESHOLD,
            cfg.MODEL.RETINANET.BG_IOU_THRESHOLD,
            allow_low_quality_matches=True,
        )

        # Loss parameters
        self.focal_loss_alpha = cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta = cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA

        self.num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
        self.in_features = cfg.MODEL.RETINANET.IN_FEATURES

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
        box_cls, box_regression = self.box_cls_and_regression(features)
        anchors = self.anchor_generator(images, features)

        if self.training:
            gt_classes, gt_anchors_reg_deltas = self.get_ground_truth(
                anchors, targets)
            return self.losses(gt_classes, gt_anchors_reg_deltas, box_cls, box_regression)

        # TODO: Remove when inference is done.
        return None

    def losses(
            self, gt_classes, gt_anchors_deltas, pred_class_logits, pred_anchor_deltas
    ):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            For `pred_class_logits` and `pred_anchor_deltas`, see
                :meth:`RetinaNetClassifierRegressionHead.forward`.

        Returns:
            losses (dict[str: Tensor]): mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_class_logits, pred_anchor_deltas = \
            _concat_box_prediction_layers(pred_class_logits, pred_anchor_deltas)
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
            reduction="sum") / max(1, num_foreground)

        # regression loss
        loss_box_reg = smooth_l1_loss(
            pred_anchor_deltas[foreground_idxs],
            gt_anchors_deltas[foreground_idxs],
            beta=self.smooth_l1_loss_beta) / max(1, num_foreground)

        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}

    def get_ground_truth(self, anchors, targets=None):
        """
        Args:
            anchors (list[list[Boxes]]): a list of #image elements. Each is a
                list of #feature level Boxes. The Boxes contains anchors of
                this image on the specific feature level.
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
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
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes,
                anchors_per_image)
            gt_matched_idxs = self.matcher(match_quality_matrix)

            # ground truth box regression
            matched_gt_boxes = targets_per_image[gt_matched_idxs.clamp(min=0)].gt_boxes
            gt_anchors_reg_deltas_i = self.box2box_transform.get_deltas(
                anchors_per_image.tensor, matched_gt_boxes.tensor)

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

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


@META_ARCH_REGISTRY.register()
class RetinaNetClassifierRegressionHead(nn.Module):
    def __init__(self, cfg):
        """
        Creates a head for object classification subnet and box regression
        subnet. Although the two subnets share a common structure, they use
        separate parameters (unlike RPN).
        """
        super(RetinaNetClassifierRegressionHead, self).__init__()

        self.in_features = ["p3", "p4", "p5", "p6", "p7"]
        in_channels = cfg.MODEL.FPN.OUT_CHANNELS
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
        # aspect ratio list ANCHOR_ASPECT_RATIOS[0] is used for all IN_FEATURES.
        assert len(cfg.MODEL.RETINANET.ANCHOR_ASPECT_RATIOS) == 1,  \
            "Using different aspect ratios between levels is not currently supported!"
        num_anchors = len(cfg.MODEL.RETINANET.ANCHOR_ASPECT_RATIOS[0]) \
            * cfg.MODEL.RETINANET.SCALES_PER_OCTAVE

        cls_subnet = []
        bbox_subnet = []
        for _ in range(cfg.MODEL.RETINANET.NUM_CONVS):
            cls_subnet.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_subnet.append(nn.ReLU())

        self.add_module('cls_subnet', nn.Sequential(*cls_subnet))
        self.add_module('bbox_subnet', nn.Sequential(*bbox_subnet))
        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1
        )

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_logits,
                        self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

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
            logits.append(self.cls_logits(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg
