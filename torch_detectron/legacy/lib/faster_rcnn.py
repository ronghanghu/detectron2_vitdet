import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from torchvision import models

from lib.utils.target_assigner import Matcher, ImageBalancedPositiveNegativeSampler, batch_box_iou
from lib.utils.box_coder import BoxCoder
from lib.utils.anchor_grid_generator import AnchorGridGenerator
import lib.utils.box as box_utils

from lib.layers import ROIAlign, ROIPool, FixedBatchNorm2d


class FasterRCNN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(FasterRCNN, self).__init__(**kwargs)
        self.num_classes = kwargs.get('num_classes')
        anchor_grid_generator = AnchorGridGenerator(16, (32, 64, 128, 256, 512), (0.5, 1, 2), max_size=1333)

        self.anchors_template = anchor_grid_generator.generate()
        if torch.cuda.is_available():
            self.anchors_template = self.anchors_template.cuda()

        # TODO this is IN TOTAL, not PER IMAGE
        batch_size = kwargs.get('batch_size')
        rpn_batch_size = kwargs.get('rpn_batch_size')
        self.rpn_only = kwargs.get('rpn_only')

        # TODO fix / improve
        self.rpn = RPN(256 * 4, anchor_grid_generator.cell_anchors.shape[0], rpn_batch_size)

        self.proposal_matcher = Matcher(0.5, 0.0)
        # fg_bg_sampler = BalancedPositiveNegativeSampler(128, 0.25)
        self.fg_bg_sampler = ImageBalancedPositiveNegativeSampler(batch_size, 0.25)

        self.box_coder = BoxCoder(weights=(10., 10., 5., 5.))

        self.selector = RPNBoxSelector(6000 * 2, 1000 * 2, 0.7, 0)

    def _generate_anchors(self, map_shape, im_shape):
        batch_size = im_shape.size(0)
        all_anchors = self.anchors_template[None]
        height, width = map_shape
        all_anchors = all_anchors[:, :height, :width]
        _, height, width, num_anchors, _ = all_anchors.shape
        all_anchors = all_anchors.expand(batch_size, *all_anchors.shape[1:])
        img_idx_proposal_full = torch.arange(batch_size, dtype=torch.int64)[:, None, None, None].expand(*all_anchors.shape[:-1])# .contiguous().view(-1)

        if im_shape.is_cuda:
            all_anchors = all_anchors.cuda()
            img_idx_proposal_full = img_idx_proposal_full.cuda()

        # TODO make this line more understandabla
        # im_height, im_width = tuple(t.squeeze(1) for t in im_shape[img_idx_proposal_full].split(1, dim=1))
        proposal_img_shapes = im_shape[img_idx_proposal_full]

        straddle_thresh = 0
        # TODO  remove nonzero and keep the binary mask, on the original shape
        inds_inside = (
            (all_anchors[..., 0] >= -straddle_thresh) &
            (all_anchors[..., 1] >= -straddle_thresh) &
            (all_anchors[..., 2] < proposal_img_shapes[..., 1] + straddle_thresh) &
            (all_anchors[..., 3] < proposal_img_shapes[..., 0] + straddle_thresh)
        )
        return all_anchors, img_idx_proposal_full, inds_inside

    def forward(self, imgs, im_shape, gt_boxes=None, img_idx_gt=None, gt_labels=None):
        # im_shape = [im.shape[1:] for im in imgs]
        features = self._backbone(imgs)
        rpn_scores, rpn_box_deltas = self.rpn(features)

        map_shape = rpn_scores.shape[-2:]
        all_anchors, img_idx_proposal, inds_inside = self._generate_anchors(map_shape, im_shape)
        if self.rpn_only:
            if gt_boxes is not None:
                rpn_labels, rpn_box_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = self.rpn.prepare_targets(all_anchors, gt_boxes, img_idx_proposal, img_idx_gt, inds_inside)
                return rpn_scores, rpn_box_deltas, rpn_labels, rpn_box_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
            return rpn_scores, rpn_box_deltas, all_anchors, img_idx_proposal, inds_inside

        with torch.no_grad():
            boxes, selected_scores, idxs = self.selector.select_top(all_anchors, F.sigmoid(rpn_scores), rpn_box_deltas, im_shape)

        rois = torch.cat([idxs[:, None].float(), boxes], dim=1)

        if gt_boxes is not None:
            rpn_labels, rpn_box_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = self.rpn.prepare_targets(all_anchors, gt_boxes, img_idx_proposal, img_idx_gt, inds_inside)
            labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, idxs_, num_pos  = self.prepare_targets(boxes, gt_boxes, idxs, img_idx_gt, gt_labels)
            rois = rois[idxs_]

        pooled_img = self._roi_pool(features, rois)
        scores, box_deltas = self._classifier(pooled_img)

        if gt_boxes is not None:
            return scores, box_deltas, rpn_scores, rpn_box_deltas, rpn_labels, rpn_box_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, num_pos

        return scores, box_deltas, rpn_scores, rpn_box_deltas, boxes, all_anchors, img_idx_proposal, inds_inside, idxs


    def prepare_targets(self, proposals, gt_boxes, img_idx_proposal, img_idx_gt, gt_labels):

        # TODO maybe make this a list of matrices if this is too big
        match_quality_matrix = batch_box_iou(gt_boxes, proposals, img_idx_gt, img_idx_proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        sampled_idxs, num_pos = self.fg_bg_sampler(matched_idxs, img_idx_proposal)

        num_proposals = sampled_idxs.size(0)

        sampled_proposals = proposals[sampled_idxs]
        # TODO replace with torch.where once Tensors and Variables are merged? Maybe not
        labels = proposals.new(num_proposals).zero_()
        labels[:num_pos] = gt_labels[matched_idxs[sampled_idxs[:num_pos]]]
        labels = labels.long()
        img_idx = img_idx_proposal[sampled_idxs]

        # get bbox regression targets
        num_classes = self.num_classes

        aligned_gt_boxes = gt_boxes[matched_idxs[sampled_idxs[:num_pos]]]
        if num_pos > 0:
            box_targets = self.box_coder.encode(aligned_gt_boxes, sampled_proposals[:num_pos])
        else:
            box_targets = aligned_gt_boxes.new()

        def expand_bbox_targets(box_targets, labels, num_classes):
            num_proposals = labels.size(0)
            expanded_box_targets = torch.zeros(num_proposals * num_classes, 4, dtype=box_targets.dtype)
            if num_pos > 0:
                map_box_idxs = labels[:num_pos] + torch.arange(num_pos, dtype=labels.dtype) * num_classes  # TODO add step instead of multiply in arange
            else:
                map_box_idxs = labels.new()
            
            expanded_box_targets[map_box_idxs] = box_targets
            expanded_box_targets = expanded_box_targets.view(num_proposals, num_classes * 4)

            bbox_inside_weights = torch.zeros_like(expanded_box_targets).view(-1, 4)
            bbox_inside_weights[map_box_idxs] = 1
            # bbox_inside_weights = bbox_inside_weights.view(-1, 4)
            bbox_inside_weights = bbox_inside_weights.view_as(expanded_box_targets)
            bbox_outside_weights = (bbox_inside_weights > 0).type_as(bbox_inside_weights)

            return expanded_box_targets, bbox_inside_weights, bbox_outside_weights

        expanded_box_targets, bbox_inside_weights, bbox_outside_weights = expand_bbox_targets(box_targets, labels, num_classes)

        return labels, expanded_box_targets, bbox_inside_weights, bbox_outside_weights, sampled_idxs, num_pos



from utils.cython_nms import nms as cython_nms
class RPNBoxSelector(object):
    def __init__(self, pre_nms_top_n, post_nms_top_n, nms_thresh, min_size):
        # TODO ATTENTION, as those numbers are for single-image in Detectron, and here it's for the batch
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size

        self.box_coder = BoxCoder(weights=(1., 1., 1., 1.))

    def select_top(self, anchors, rpn_scores, rpn_box_deltas, im_shapes):
        # anchors = [N, H, W, A, 4]
        # img_idx_proposals = [N, H, W, A] == inds_inside
        # rpn_scores = [N, A, H, W]
        # rpn_box_deltas = [N, A * 4, H, W]

        N, H, W, A, _ = anchors.shape

        # TODO replace by reshape when merged
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(N, -1)
        rpn_box_deltas = rpn_box_deltas.view(N, A, 4, H, W).permute(0, 3, 4, 1, 2).contiguous().view(N, -1, 4)

        rpn_scores, order = rpn_scores.topk(self.pre_nms_top_n, dim=1, sorted=True)

        # TODO attention with the GPU device
        order = (torch.arange(N, dtype=order.dtype)[:, None], order)
        rpn_box_deltas = rpn_box_deltas[order]
        anchors = anchors.contiguous().view(N, -1, 4)[order]

        proposals = self.box_coder.decode(rpn_box_deltas.view(-1, 4), anchors.view(-1, 4))

        proposals = proposals.view(N, -1, 4)

        # TODO optimize / make batch friendly
        sampled_proposals = []
        sampled_scores = []
        for proposal, score, im_shape in zip(proposals, rpn_scores, im_shapes):
            height, width = im_shape
            p = box_utils.clip_boxes_to_image(proposal, height, width)
            keep = _filter_boxes(p, self.min_size, im_shape)
            p = p[keep, :]
            score = score[keep]
            if self.nms_thresh > 0:
                # keep = box_utils.nms(p, score, self.nms_thresh)
                keep = torch.from_numpy(cython_nms(torch.cat((p, score[:, None]), 1).detach().cpu().numpy(), self.nms_thresh)).cuda()
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


class ResNet(models.ResNet):
    def __init__(self, block, layers, num_classes, **kwargs):
        super(ResNet, self).__init__(block, layers, num_classes)

class RPN(nn.Module):
    def __init__(self, inplanes, num_anchors, batch_size):
        super(RPN, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(inplanes, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(inplanes, num_anchors * 4, kernel_size=1, stride=1)
        # self.conv.weight.data.fill_(0)
        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal(l.weight, std=0.01)
            torch.nn.init.constant(l.bias, 0)

        self.proposal_matcher = Matcher(0.7, 0.3, force_match_for_each_row=True)
        self.fg_bg_sampler = ImageBalancedPositiveNegativeSampler(batch_size, 0.5, which_negatives=-1)
        self.box_coder = BoxCoder(weights=(1., 1., 1., 1.))

    def forward(self, x):
        x = F.relu(self.conv(x))
        logits = self.cls_logits(x)
        bbox_reg = self.bbox_pred(x)
        return logits, bbox_reg

    def prepare_targets(self, proposals, gt_boxes, img_idx_proposal, img_idx_gt, proposals_ids_inside):

        # TODO replace by reshape when merged
        # TODO can maybe remove the permute -- the order is maybe not necessary
        total_anchors = img_idx_proposal.numel()  # TODO improve, looks ugly
        batch_size = proposals.shape[0]
        proposals = proposals.permute(0, 3, 1, 2, 4).contiguous().view(-1, 4)
        img_idx_proposal = img_idx_proposal.permute(0, 3, 1, 2).contiguous().view(-1)
        proposals_ids_inside = proposals_ids_inside.permute(0, 3, 1, 2).contiguous().view(-1)

        proposals = proposals[proposals_ids_inside]
        img_idx_proposal = img_idx_proposal[proposals_ids_inside]

        # TODO maybe make this a list of matrices if this is too big
        match_quality_matrix = batch_box_iou(gt_boxes, proposals, img_idx_gt, img_idx_proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        sampled_idxs, num_pos = self.fg_bg_sampler(matched_idxs, img_idx_proposal)

        num_anchors = proposals.size(0)

        sampled_proposals = proposals[sampled_idxs[:num_pos]]
        # TODO replace with torch.where once Tensors and Variables are merged? Maybe not
        labels = proposals.new(num_anchors).fill_(-1)
        labels[sampled_idxs[:num_pos]] = 1
        labels[sampled_idxs[num_pos:]] = 0
        labels = labels.long()
        # img_idx = img_idx_proposal[sampled_idxs]

        aligned_gt_boxes = gt_boxes[matched_idxs[sampled_idxs[:num_pos]]]
        box_targets = self.box_coder.encode(aligned_gt_boxes, sampled_proposals)

        expanded_box_targets = proposals.new(num_anchors, 4).zero_()
        expanded_box_targets[sampled_idxs[:num_pos]] = box_targets

        bbox_inside_weights = torch.zeros_like(expanded_box_targets)
        bbox_inside_weights[sampled_idxs[:num_pos]] = 1

        bbox_outside_weights = torch.zeros_like(expanded_box_targets)
        # TODO TODO TODO TODO CHECK THIS FIXME FIXME looks like we should remove batch_size??? FIXME
        # bbox_outside_weights[sampled_idxs] = 1.0 / (sampled_idxs.shape[0] * batch_size)  ## TODO looks like they divide by the number of ROIs PER IMAGE, not global THIS IS AN APPROXIMATION!
        bbox_outside_weights[sampled_idxs] = 1.0 / (sampled_idxs.shape[0]) * batch_size  ## TODO looks like they divide by the number of ROIs PER IMAGE, not global THIS IS AN APPROXIMATION!

        # unmap to original ids
        exp_labels = labels.new(total_anchors).fill_(-1)
        exp_labels[proposals_ids_inside] = labels
        exp_box_targets = expanded_box_targets.new(total_anchors, 4).fill_(0)
        exp_box_targets[proposals_ids_inside] = expanded_box_targets
        exp_bbox_inside_weights = bbox_inside_weights.new(total_anchors, 4).fill_(0)
        exp_bbox_inside_weights[proposals_ids_inside] = bbox_inside_weights
        exp_bbox_outside_weights = bbox_outside_weights.new(total_anchors, 4).fill_(0)
        exp_bbox_outside_weights[proposals_ids_inside] = bbox_outside_weights


        # return labels, expanded_box_targets, bbox_inside_weights, bbox_outside_weights
        return exp_labels, exp_box_targets, exp_bbox_inside_weights, exp_bbox_outside_weights




class ResNetFasterRCNN(FasterRCNN, ResNet):
    def __init__(self, block, layers, num_classes):
        super(ResNetFasterRCNN, self).__init__(block=block, layers=layers, num_classes=1000)

        self._roi_pool = ROIAlign(output_size=(14, 14),
                spatial_scale=1.0 / 16, sampling_ratio=0)
        self.cls_score = nn.Linear(512 * block.expansion, num_classes)
        self.bbox_pred = nn.Linear(512 * block.expansion, num_classes * 4)

        nn.init.normal(self.cls_score.weight, std=0.01)
        nn.init.constant(self.cls_score.bias, 0)
        nn.init.normal(self.bbox_pred.weight, std=0.001)
        nn.init.constant(self.bbox_pred.bias, 0)

    def _backbone(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = x.detach()

        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def _classifier(self, x):
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


from lib.models_caffe2.fastrcnn_resnet50c4_1x import ResNet50_FastRCNN
class C2ResNetFasterRCNN(FasterRCNN, ResNet50_FastRCNN):
    def __init__(self, weights=None, num_classes=81, rpn_batch_size=None, batch_size=None, rpn_only=None):
        super(C2ResNetFasterRCNN, self).__init__(num_classes=num_classes, rpn_batch_size=batch_size, batch_size=batch_size, rpn_only=rpn_only)
        if weights is not None:
            import pickle
            if isinstance(weights, str):  # TODO py2 compat
                with open(weights, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                weights = data['blobs']
            weights = self._parse_weights(weights)
            self.load_state_dict(weights, strict=False)




def fasterrcnn_resnet18(**kwargs):
    model = ResNetFasterRCNN(models.resnet.BasicBlock, [2, 2, 2, 2], **kwargs)
    model_urls = models.resnet.model_urls
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    # _fix_batchnorm2d(model)
    # print(model)
    return model
