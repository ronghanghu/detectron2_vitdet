import torch
import math


class BoxCoder(object):

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, boxes, anchors):
        """
        Arguments:
            boxes: tensor
            anchors: tensor
        """
        TO_REMOVE = 1  # TODO remove
        ex_widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
        ex_heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
        ex_ctr_x = anchors[:, 0] + 0.5 * ex_widths
        ex_ctr_y = anchors[:, 1] + 0.5 * ex_heights

        gt_widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        gt_heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        gt_ctr_x = boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = boxes[:, 1] + 0.5 * gt_heights

        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)

        targets = torch.stack((targets_dx, targets_dy, targets_dw,
                              targets_dh), dim=1)
        return targets

    def decode(self, rel_codes, anchors):
        """
        Arguments:
            boxes: tensor
            anchors: tensor
        """
        if anchors.shape[0] == 0:
            # TODO this might crash
            # return torch.zeros(0, rel_codes.shape[1], dtype=rel_codes.dtype)
            return rel_codes.new_zeros(0, rel_codes.shape[1])

        anchors = anchors.type(rel_codes.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
        heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # pred_boxes = torch.zeros(rel_codes.shape, dtype=rel_codes.dtype)
        pred_boxes = torch.zeros_like(rel_codes)
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

        return pred_boxes


