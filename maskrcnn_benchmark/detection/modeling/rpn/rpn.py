import torch
import torch.nn.functional as F
from torch import nn

from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from .anchor_generator import build_anchor_generator
from .rpn_outputs import RPNOutputs


class RPNHead(nn.Module):
    """
    RPN classification and regression heads. Uses a 3x3 conv to produce a shared
    hidden state from which one 1x1 conv predicts objectness scores for each anchor
    and a second 1x1 conv predicts bounding-box deltas specifying how to deform
    each anchor into an object proposal.
    """

    def __init__(self, in_channels, num_cell_anchors):
        """
        Args:
            in_channels (int): number of channels of the input feature map
            num_cell_anchors (int): number of cell anchors
        """
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_cell_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_cell_anchors * 4, kernel_size=1, stride=1)

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of feature maps
        """
        cls_logits = []
        bbox_pred = []
        for x in features:
            t = F.relu(self.conv(x))
            cls_logits.append(self.cls_logits(t))
            bbox_pred.append(self.bbox_pred(t))
        return cls_logits, bbox_pred


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
        self.min_size               = cfg.MODEL.RPN.MIN_SIZE
        self.batch_size_per_image   = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction      = cfg.MODEL.RPN.POSITIVE_FRACTION
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

        self.anchor_generator = build_anchor_generator(cfg)
        self.box2box_transform = Box2BoxTransform(weights=(1.0, 1.0, 1.0, 1.0))
        self.anchor_matcher = Matcher(
            cfg.MODEL.RPN.FG_IOU_THRESHOLD,
            cfg.MODEL.RPN.BG_IOU_THRESHOLD,
            allow_low_quality_matches=True,
        )
        self.head = self._build_rpn_head(cfg)

    def _build_rpn_head(self, cfg):
        """
        When RPN is applied on multiple feature maps (as in FPN), we share the
        same RPNHead and therefore the channel counts and number of cell anchors
        must be the same. Check that this is true and return the counts.
        """
        feature_channels = dict(
            zip(cfg.MODEL.BACKBONE.OUT_FEATURES, cfg.MODEL.BACKBONE.OUT_FEATURE_CHANNELS)
        )
        in_channels = [feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        for c in in_channels:
            assert c == in_channels[0]

        num_cell_anchors = self.anchor_generator.num_cell_anchors
        # Check all cell anchor counts are equal
        for c in num_cell_anchors:
            assert c == num_cell_anchors[0]

        return RPNHead(in_channels[0], num_cell_anchors[0])

    def forward(self, images, features, targets=None):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            targets (list[BoxList, optional): length `N` list of `BoxList`s. Each
                `BoxList` stores ground-truth boxes for the corresponding image.
        """
        features = [features[f] for f in self.in_features]
        objectness, anchor_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)
        # TODO: The anchors only depend on the feature map shape; there's probably
        # an opportunity for some optimizations (e.g., caching anchors).
        outputs = RPNOutputs(
            self.box2box_transform,
            self.anchor_matcher,
            self.batch_size_per_image,
            self.positive_fraction,
            images,
            objectness,
            anchor_deltas,
            anchors,
            targets,
        )

        if self.training:
            losses = outputs.losses()
            if self.rpn_only:
                # When training an RPN-only model, the loss is determined by the
                # predicted objectness and anchor_deltas values and there is
                # no need to transform the anchors into predicted boxes; this is an
                # optimization that avoids the unnecessary transformation.
                return None, losses
        else:
            losses = {}

        with torch.no_grad():
            proposals = outputs.proposals(
                nms_thresh=self.nms_thresh,
                pre_nms_topk=self.pre_nms_topk[self.training],
                post_nms_topk=self.post_nms_topk[self.training],
                min_size=self.min_size,
                training=self.training,
            )

            # Augment RPN proposals with ground-truth boxes
            if self.training:
                proposals = outputs.add_gt_proposals(proposals)

            if self.rpn_only:
                # We must be in testing mode (self.training and self.rpn_only returns early)
                assert not self.training
                # For end-to-end models, the RPN proposals are an intermediate state
                # and don't bother to sort them in decreasing score order. For RPN-only
                # models, the proposals are the final output and we return them in
                # high-to-low confidence order.
                inds = [box.get_field("objectness").sort(descending=True)[1] for box in proposals]
                proposals = [box[ind] for box, ind in zip(proposals, inds)]

        return proposals, losses


def build_rpn(cfg):
    return RPN(cfg)
