import math
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.structures import Instances
from detectron2.utils.registry import Registry

from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from .anchor_generator import build_anchor_generator
from .rpn_outputs import RPNOutputs, find_top_rpn_proposals


# In the future when we have more proposal mechanisms (including precomputed)
# this function should move into a shared module.
def add_ground_truth_to_proposals(gt_boxes, proposals):
    """
    Augment `proposals` with ground-truth boxes from `gt_boxes`.

    Args:
        gt_boxes(list[Boxes]): list of N elements. Element i is a Boxes
            representing the gound-truth for image i.
        proposals (list[Instances]): list of N elements. Element i is a Instances
            representing the proposals for image i.

    Returns:
        list[Instances]: list of N Instances. Each is the proposals for the image,
            with field "proposal_boxes" and "objectness_logits".
    """
    assert gt_boxes is not None

    assert len(proposals) == len(gt_boxes)
    if len(proposals) == 0:
        return proposals

    device = proposals[0].objectness_logits.device
    # Concating gt_boxes with proposals requires them to have the same fields
    # Assign all ground-truth boxes an objectness logit corresponding to P(object) \approx 1.
    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))

    new_proposals = []
    for proposal, gt_boxes_i in zip(proposals, gt_boxes):
        gt_logits_i = gt_logit_value * torch.ones(len(gt_boxes_i), device=device)
        gt_proposal = Instances(proposal.image_size)

        gt_proposal.proposal_boxes = gt_boxes_i
        gt_proposal.objectness_logits = gt_logits_i
        new_proposals.append(Instances.cat([proposal, gt_proposal]))
    return new_proposals


RPN_HEAD_REGISTRY = Registry("RPN_HEAD")


def build_rpn_head(cfg):
    name = cfg.MODEL.RPN_HEAD.NAME
    return RPN_HEAD_REGISTRY.get(name)(cfg)


def shared_rpn_head_in_channels(cfg):
    """
    This function calculates the number of in channels of RPN Head and is
    specific to the case in which the RPN head shares conv layers over all
    feature map levels. That's why there's a check that all channel counts
    are the same.
    """
    in_features = cfg.MODEL.RPN.IN_FEATURES
    feature_channels = dict(cfg.MODEL.BACKBONE.COMPUTED_OUT_FEATURE_CHANNELS)
    in_feature_channels = [feature_channels[f] for f in in_features]
    # Check all channel counts are equal
    for c in in_feature_channels:
        assert c == in_feature_channels[0]
    return in_feature_channels[0]


def shared_rpn_head_num_cell_anchors(cfg):
    """
    This function calculates the number of anchor cells. This is specific to
    a RPN Head which shares conv layers over all feature map levels.
    That's why the length of index [0] for anchor sizes and aspect ratios is
    correct (even though it ignores the other indices).
    """
    anchor_sizes = cfg.MODEL.RPN.ANCHOR_SIZES
    # Check that all levels use the same number of anchor sizes
    if len(anchor_sizes) > 1:
        for c in anchor_sizes:
            assert len(c) == len(anchor_sizes[0])
    anchor_aspect_ratios = cfg.MODEL.RPN.ANCHOR_ASPECT_RATIOS
    # Check that all levels use the same number of aspect ratios
    if len(anchor_aspect_ratios) > 1:
        for c in anchor_aspect_ratios:
            assert len(c) == len(anchor_aspect_ratios[0])
    return len(anchor_sizes[0]) * len(anchor_aspect_ratios[0])


@RPN_HEAD_REGISTRY.register()
class StandardRPNHead(nn.Module):
    """
    RPN classification and regression heads. Uses a 3x3 conv to produce a shared
    hidden state from which one 1x1 conv predicts objectness logits for each anchor
    and a second 1x1 conv predicts bounding-box deltas specifying how to deform
    each anchor into an object proposal.
    """

    def __init__(self, cfg):
        super(StandardRPNHead, self).__init__()

        in_channels = shared_rpn_head_in_channels(cfg)
        num_cell_anchors = shared_rpn_head_num_cell_anchors(cfg)

        # 3x3 conv for the hidden representation
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(in_channels, num_cell_anchors, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(in_channels, num_cell_anchors * 4, kernel_size=1, stride=1)

        for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of feature maps
        """
        objectness_logits_pred = []
        anchor_deltas_pred = []
        for x in features:
            t = F.relu(self.conv(x))
            objectness_logits_pred.append(self.objectness_logits(t))
            anchor_deltas_pred.append(self.anchor_deltas(t))
        return objectness_logits_pred, anchor_deltas_pred


class RPN(nn.Module):
    """
    RPN subnetwork.
    """

    def __init__(self, cfg):
        super(RPN, self).__init__()

        # fmt: off
        self.rpn_only               = cfg.MODEL.RPN_ONLY
        self.in_features            = cfg.MODEL.RPN.IN_FEATURES
        self.nms_thresh             = cfg.MODEL.RPN.NMS_THRESH
        self.min_box_side_len       = cfg.MODEL.RPN.MIN_SIZE
        self.batch_size_per_image   = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction      = cfg.MODEL.RPN.POSITIVE_FRACTION
        self.smooth_l1_beta         = cfg.MODEL.RPN.SMOOTH_L1_BETA
        # fmt: on

        # Map from self.training state to train/test settings
        self.pre_nms_topk = {
            True: cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.PRE_NMS_TOPK_TEST,
        }
        self.post_nms_topk = {
            True: cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.POST_NMS_TOPK_TEST,
        }
        self.boundary_threshold = cfg.MODEL.RPN.BOUNDARY_THRESH

        self.anchor_generator = build_anchor_generator(cfg)
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(
            cfg.MODEL.RPN.FG_IOU_THRESHOLD,
            cfg.MODEL.RPN.BG_IOU_THRESHOLD,
            allow_low_quality_matches=True,
        )
        self.head = build_rpn_head(cfg)

    def forward(self, images, features, gt_instances=None):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s. Each
                `Instances` stores ground-truth instances for the corresponding image.
        """
        gt_boxes = [x.gt_boxes for x in gt_instances] if gt_instances is not None else None
        del gt_instances
        features = [features[f] for f in self.in_features]
        objectness_logits_pred, anchor_deltas_pred = self.head(features)
        anchors = self.anchor_generator(images, features)
        # TODO: The anchors only depend on the feature map shape; there's probably
        # an opportunity for some optimizations (e.g., caching anchors).
        outputs = RPNOutputs(
            self.box2box_transform,
            self.anchor_matcher,
            self.batch_size_per_image,
            self.positive_fraction,
            images,
            objectness_logits_pred,
            anchor_deltas_pred,
            anchors,
            self.boundary_threshold,
            gt_boxes,
            self.smooth_l1_beta,
        )

        if self.training:
            losses = outputs.losses()
            if self.rpn_only:
                # When training an RPN-only model, the loss is determined by the
                # objectness_logits_pred and anchor_deltas_pred values and there is
                # no need to transform the anchors into predicted boxes; this is an
                # optimization that avoids the unnecessary transformation.
                return None, losses
        else:
            losses = {}

        with torch.no_grad():
            # Find the top proposals by applying NMS and removing boxes that
            # are too small. The proposals are treated as fixed for approximate
            # joint training with roi heads. This approach ignores the derivative
            # w.r.t. the proposal boxes’ coordinates that are also network
            # responses, so is approximate.
            proposals = find_top_rpn_proposals(
                outputs.predict_proposals(),
                outputs.predict_objectness_logits(),
                images,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_side_len,
                self.training,
            )

            if self.training:
                # Augment RPN proposals with ground-truth boxes.
                # When training starts, RPN will produce low quality proposals due
                # to random initialization. It's possible that none of these initial
                # proposals have high enough overlap with the gt objects to be used
                # as positive examples for the second stage components (box head,
                # cls head, mask head). Adding the gt boxes to the set of proposals
                # ensures that the second stage components will have some positive
                # examples from the start of training. This augmentation improves
                # convergence and empirically improves box AP on COCO by about 0.5
                # points (under one tested configuration).
                proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

            if self.rpn_only:
                # We must be in testing mode (self.training and self.rpn_only returns early)
                assert not self.training
                # For end-to-end models, the RPN proposals are an intermediate state
                # and don't bother to sort them in decreasing score order. For RPN-only
                # models, the proposals are the final output and we return them in
                # high-to-low confidence order.
                inds = [p.objectness_logits.sort(descending=True)[1] for p in proposals]
                proposals = [p[ind] for p, ind in zip(proposals, inds)]

        return proposals, losses


def build_rpn(cfg):
    return RPN(cfg)
