import torch
from torch import nn
import torch.nn.functional as F

from box_coder import BoxCoder
from anchor_grid_generator import AnchorGridGenerator
import box as box_utils

from nms import nms


class FasterRCNN(torch.nn.Module):
    def __init__(self, rpn, **kwargs):
        super(FasterRCNN, self).__init__(**kwargs)
        self.rpn_only = kwargs.get('rpn_only')
        self.rpn = rpn

    def forward(self, imgs, im_shape):
        features = self._backbone(imgs)
        rpn_scores, rpn_box_deltas = self.rpn(features)

        map_shape = rpn_scores.shape[-2:]
        all_anchors, img_idx_proposal, inds_inside = self.rpn._generate_anchors(
                map_shape, im_shape)
        if self.rpn_only:
            return rpn_scores, rpn_box_deltas, all_anchors, img_idx_proposal, inds_inside

        with torch.no_grad():
            boxes, selected_scores, idxs = self.rpn.selector.select_top(
                    all_anchors, F.sigmoid(rpn_scores), rpn_box_deltas, im_shape)

        rois = torch.cat([idxs[:, None].float(), boxes], dim=1)
        rois = rois.cuda()

        pooled_img = self._roi_pool(features, rois)
        scores, box_deltas = self._classifier(pooled_img)

        return scores, box_deltas, rpn_scores, rpn_box_deltas, boxes, all_anchors, img_idx_proposal, inds_inside, idxs


class RPNBoxSelector(object):
    def __init__(self, pre_nms_top_n, post_nms_top_n, nms_thresh, min_size,
            box_coder = BoxCoder(weights=(1., 1., 1., 1.))):
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size

        self.box_coder = box_coder

    def select_top(self, anchors, rpn_scores, rpn_box_deltas, im_shapes):
        N, H, W, A, _ = anchors.shape

        rpn_scores = rpn_scores.permute(0, 2, 3, 1).reshape(N, -1)
        rpn_box_deltas = rpn_box_deltas.view(N, A, 4, H, W).permute(
                0, 3, 4, 1, 2).reshape(N, -1, 4)

        rpn_scores, order = rpn_scores.topk(self.pre_nms_top_n, dim=1, sorted=True)

        # TODO attention with the GPU device
        order = (torch.arange(N, dtype=order.dtype)[:, None], order)
        rpn_box_deltas = rpn_box_deltas[order]
        anchors = anchors.contiguous().view(N, -1, 4)[order]

        proposals = self.box_coder.decode(
                rpn_box_deltas.view(-1, 4), anchors.view(-1, 4))

        proposals = proposals.view(N, -1, 4)

        sampled_proposals = []
        sampled_scores = []
        for proposal, score, im_shape in zip(proposals, rpn_scores, im_shapes):
            height, width = im_shape
            p = box_utils.clip_boxes_to_image(proposal, height, width)
            keep = _filter_boxes(p, self.min_size, im_shape)
            p = p[keep, :]
            score = score[keep]
            if self.nms_thresh > 0:
                keep = nms(p.cpu(), score.cpu(), self.nms_thresh)
                if self.post_nms_top_n > 0:
                    keep = keep[:self.post_nms_top_n]
                p = p[keep]
                score = score[keep]
            sampled_proposals.append(p)
            sampled_scores.append(score)

        ids = [i for i, x in enumerate(sampled_proposals) for _ in range(len(x))]
        ids = torch.tensor(ids, dtype=order[0].dtype)
        sampled_proposals = torch.cat(sampled_proposals, dim=0)

        return sampled_proposals, sampled_scores, ids


def _filter_boxes(boxes, min_size, im_shape):
    """Only keep boxes with both sides >= min_size and center within the image.
    """
    # Scale min_size to match image scale
    fact = 1  # TODO remove
    ws = boxes[:, 2] - boxes[:, 0] + fact
    hs = boxes[:, 3] - boxes[:, 1] + fact
    x_ctr = boxes[:, 0] + ws / 2.
    y_ctr = boxes[:, 1] + hs / 2.
    keep = torch.nonzero(
        (ws >= min_size) & (hs >= min_size) &
        (x_ctr < im_shape[1]) & (y_ctr < im_shape[0])).squeeze(1)
    return keep


class RPN(nn.Module):
    def __init__(self, inplanes,
            anchor_grid_generator=AnchorGridGenerator(
                16, (32, 64, 128, 256, 512), (0.5, 1, 2), max_size=1333),
            selector=RPNBoxSelector(6000, 1000, 0.7, 0)):
        super(RPN, self).__init__()

        num_anchors = anchor_grid_generator.cell_anchors.shape[0]

        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(inplanes, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(inplanes, num_anchors * 4, kernel_size=1, stride=1)

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal(l.weight, std=0.01)
            torch.nn.init.constant(l.bias, 0)

        self.register_buffer('anchors_template', anchor_grid_generator.generate())

        self.selector = selector

    def forward(self, x):
        x = F.relu(self.conv(x))
        logits = self.cls_logits(x)
        bbox_reg = self.bbox_pred(x)
        return logits, bbox_reg

    def _generate_anchors(self, map_shape, im_shape):
        batch_size = im_shape.size(0)
        all_anchors = self.anchors_template[None]
        height, width = map_shape
        all_anchors = all_anchors[:, :height, :width]
        _, height, width, num_anchors, _ = all_anchors.shape
        all_anchors = all_anchors.expand(batch_size, *all_anchors.shape[1:])
        img_idx_proposal_full = torch.arange(batch_size, dtype=torch.int64)[:, None, None, None].expand(*all_anchors.shape[:-1])

        if im_shape.is_cuda:
            all_anchors = all_anchors.cuda()
            img_idx_proposal_full = img_idx_proposal_full.cuda()

        proposal_img_shapes = im_shape[img_idx_proposal_full]
        device = all_anchors.device
        proposal_img_shapes = proposal_img_shapes.to(device)

        straddle_thresh = 0
        inds_inside = (
            (all_anchors[..., 0] >= -straddle_thresh) &
            (all_anchors[..., 1] >= -straddle_thresh) &
            (all_anchors[..., 2] < proposal_img_shapes[..., 1] + straddle_thresh) &
            (all_anchors[..., 3] < proposal_img_shapes[..., 0] + straddle_thresh)
        )
        return all_anchors, img_idx_proposal_full, inds_inside

