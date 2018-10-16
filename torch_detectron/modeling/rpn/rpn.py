import torch
import torch.nn.functional as F
from torch import nn

from torch_detectron.modeling.box_coder import BoxCoder
from ..model_builder.loss_evaluators import make_rpn_loss_evaluator
from ..backbone.fpn import ROI2FPNLevelMapper
from .anchor_generator import AnchorGenerator
from .anchor_generator import FPNAnchorGenerator
from .inference import FPNRPNBoxSelector
from .inference import RPNBoxSelector


def make_anchor_generator(config):
    use_fpn = config.MODEL.RPN.USE_FPN

    scales = config.MODEL.RPN.SCALES
    aspect_ratios = config.MODEL.RPN.ASPECT_RATIOS
    base_anchor_size = config.MODEL.RPN.BASE_ANCHOR_SIZE
    anchor_stride = config.MODEL.RPN.ANCHOR_STRIDE
    straddle_thresh = config.MODEL.RPN.STRADDLE_THRESH

    anchor_maker = AnchorGenerator if not use_fpn else FPNAnchorGenerator
    anchor_args = {}
    # FIXME unify the args of AnchorGenerator and FPNAnchorGenerator?
    anchor_args["scales"] = scales
    anchor_args["aspect_ratios"] = aspect_ratios
    anchor_args["base_anchor_size"] = base_anchor_size
    anchor_args["straddle_thresh"] = straddle_thresh
    if use_fpn:
        anchor_args["anchor_strides"] = anchor_stride
        assert len(anchor_stride) == len(
            scales
        ), "FPN should have len(ANCHOR_STRIDE) == len(SCALES)"
    else:
        assert len(anchor_stride) == 1, "Non-FPN should have a single ANCHOR_STRIDE"
        anchor_args["anchor_stride"] = anchor_stride[0]
    anchor_generator = anchor_maker(**anchor_args)
    return anchor_generator


def make_box_selector(config, rpn_box_coder, is_train):
    use_fpn = config.MODEL.RPN.USE_FPN
    box_selector_maker = RPNBoxSelector
    box_selector_args = {}
    if use_fpn:
        # TODO expose those options
        roi_to_fpn_level_mapper = ROI2FPNLevelMapper(2, 5)
        box_selector_maker = FPNRPNBoxSelector
        box_selector_args["roi_to_fpn_level_mapper"] = roi_to_fpn_level_mapper
        fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN
        if not is_train:
            fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST
        box_selector_args["fpn_post_nms_top_n"] = fpn_post_nms_top_n

    pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TRAIN
    post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TRAIN
    if not is_train:
        pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TEST
        post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TEST
    nms_thresh = config.MODEL.RPN.NMS_THRESH
    min_size = config.MODEL.RPN.MIN_SIZE
    box_selector = box_selector_maker(
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        box_coder=rpn_box_coder,
        **box_selector_args
    )
    return box_selector


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

        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        box_selector_train = make_box_selector(cfg, rpn_box_coder, is_train=True)
        box_selector_test = make_box_selector(cfg, rpn_box_coder, is_train=False)

        loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
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
        anchors = self.anchor_generator(images.image_sizes, features)

        if self.training:
            return self._forward_train(anchors, objectness, rpn_box_regression, targets)
        else:
            return self._forward_test(anchors, objectness, rpn_box_regression)

    def _forward_train(self, anchors, objectness, rpn_box_regression, targets):
        if self.cfg.MODEL.RPN_ONLY:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            boxes = anchors
        else:
            # For end-to-end models, anchors must be transformed into boxes and
            # sampled into a training batch.
            with torch.no_grad():
                boxes = self.box_selector_train(
                    anchors, objectness, rpn_box_regression, targets
                )
        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
            anchors, objectness, rpn_box_regression, targets
        )
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
        return boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        if self.cfg.MODEL.RPN_ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return boxes, {}


def build_rpn(cfg):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    return RPNModule(cfg)
