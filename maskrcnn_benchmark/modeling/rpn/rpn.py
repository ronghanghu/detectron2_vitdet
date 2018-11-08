import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from .loss import make_rpn_loss_evaluator
from .anchor_generator import make_anchor_generator

from .proposals import generate_rpn_proposals, generate_fpn_proposals
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist


class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, in_channels, num_anchors):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg):
        super(RPNModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator(cfg)

        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        head = RPNHead(in_channels, anchor_generator.num_anchors_per_location()[0])

        self.rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        loss_evaluator = make_rpn_loss_evaluator(cfg, self.rpn_box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)
        """
        objectness, rpn_box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)
        # Note that anchors returned for each image have identical boxes.
        # They are only different in image shapes. TODO Opportunities for simplification.
        """
        anchors: A list of #img elements. Each element is a list of #lvl BoxList.
        objectness: A list of #lvl elements. Each element is a tensor of shape (B, #anchor, H, W)
        rpn_box_regression: A list of #lvl elements. Each element is a tensor of shape (B, #anchor x 4, H, W)
        """
        if self.training:
            loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
                anchors, objectness, rpn_box_regression, targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        else:
            losses = {}

        if self.cfg.MODEL.RPN_ONLY and self.training:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            return None, losses

        with torch.no_grad():
            decoded_boxes = []
            """
            decoded_boxes: #lvl elements. Each is a tensor of shape (B, #total anchors in the image, 4)
            """
            anchors = list(zip(*anchors))
            # for each feature level
            for anchor, bbreg_logit in zip(anchors, rpn_box_regression):
                B, _, H, W = bbreg_logit.shape
                bbreg_logit = bbreg_logit.view(B, -1, 4, H, W).permute(0, 3, 4, 1, 2).reshape(-1, 4)
                # concat all anchors over all images
                anchors = torch.cat([a.bbox for a in anchor], dim=0).reshape(-1, 4)
                decoded = self.rpn_box_coder.decode(bbreg_logit, anchors)
                decoded_boxes.append(decoded.view(B, -1, 4))   # each has shape (B, HxWx#anchor, 4)

            for idx, score in enumerate(objectness):
                objectness[idx] = score.permute(0, 2, 3, 1).reshape(B, -1)   # reshape to (B, HxWx#anchor)

            rpn_cfg = self.cfg.MODEL.RPN
            if len(features) == 1:
                proposals = generate_rpn_proposals(
                    decoded_boxes[0], objectness[0],
                    [x[::-1] for x in images.image_sizes],  # TODO The w, h order is not consistent everywhere!
                    nms_thresh=rpn_cfg.NMS_THRESH,
                    pre_nms_topk=rpn_cfg.PRE_NMS_TOP_N_TRAIN if self.training else rpn_cfg.PRE_NMS_TOP_N_TEST,
                    post_nms_topk=rpn_cfg.POST_NMS_TOP_N_TRAIN if self.training else rpn_cfg.POST_NMS_TOP_N_TEST,
                    min_size=rpn_cfg.MIN_SIZE)
            else:
                proposals = generate_fpn_proposals(
                    decoded_boxes, objectness,
                    [x[::-1] for x in images.image_sizes],  # TODO same here
                    nms_thresh=rpn_cfg.NMS_THRESH,
                    pre_nms_topk=rpn_cfg.PRE_NMS_TOP_N_TRAIN if self.training else rpn_cfg.PRE_NMS_TOP_N_TEST,
                    post_nms_topk=rpn_cfg.FPN_POST_NMS_TOP_N_TRAIN if self.training else rpn_cfg.FPN_POST_NMS_TOP_N_TEST,
                    min_size=rpn_cfg.MIN_SIZE, training=self.training)

            # Add GT proposals for end-to-end training.
            if self.training and targets is not None:
                proposals = self._add_gt_proposals(proposals, targets)

            if self.cfg.MODEL.RPN_ONLY:
                # We must be in testing mode.
                # For end-to-end models, the RPN proposals are an intermediate state
                # and don't bother to sort them in decreasing score order. For RPN-only
                # models, the proposals are the final output and we return them in
                # high-to-low confidence order.
                inds = [
                    box.get_field("objectness").sort(descending=True)[1] for box in proposals
                ]
                proposals = [box[ind] for box, ind in zip(proposals, inds)]
            return proposals, losses

    def _add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        # Get the device we're operating on
        device = proposals[0].bbox.device

        gt_boxes = [target.copy_with_fields([]) for target in targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.add_field("objectness", torch.ones(len(gt_box), device=device))

        proposals = [
            cat_boxlist((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals


def build_rpn(cfg):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    return RPNModule(cfg)
