import torch
from torch import nn
from torch.nn import functional as F

from lib.utils.target_assigner import Matcher, ImageBalancedPositiveNegativeSampler, batch_box_iou
from .utils.box_coder import BoxCoder
from .utils.anchor_grid_generator import AnchorGridGenerator
from .utils.box import boxes_area
import lib.utils.box as box_utils

from lib.utils.losses import smooth_l1_loss

from .layers import roi_align

from .collate_fn import join_targets


class FPN(nn.Module):
    def __init__(self, **kwargs):
        super(FPN, self).__init__()
        self.num_classes = kwargs.get('num_classes')

        batch_size = kwargs.get('batch_size')
        rpn_batch_size = kwargs.get('rpn_batch_size')

        self.proposal_matcher = Matcher(0.5, 0.0)
        # fg_bg_sampler = BalancedPositiveNegativeSampler(128, 0.25)
        self.fg_bg_sampler = ImageBalancedPositiveNegativeSampler(batch_size, 0.25)

        self.box_coder = BoxCoder(weights=(10., 10., 5., 5.))

        self.rpn = RPN_FPN(256 * 1, rpn_batch_size)  # TODO fix


    def losses(self, images, targets):
        (scores, box_deltas, rpn_scores, rpn_box_deltas, rpn_labels, rpn_box_targets,
        rpn_bbox_inside_weights, rpn_bbox_outside_weights, labels, bbox_targets,
        bbox_inside_weights, bbox_outside_weights, num_pos) = self(
                images, targets)

        if True:  # is not only RPN
            loss_classif = F.cross_entropy(scores, labels, size_average=True)
            loss_box_reg = smooth_l1_loss(box_deltas, bbox_targets, bbox_inside_weights, bbox_outside_weights, 1) / box_deltas.shape[0]

        rpn_classif_losses = []
        rpn_losses_box_reg = []
        for lvl in range(len(rpn_scores)):
            N, A, H, W = rpn_scores[lvl].shape
            rpn_score = rpn_scores[lvl].contiguous().view(-1)
            rpn_box_delta = rpn_box_deltas[lvl].view(N, A, 4, H, W).permute(0, 1, 3, 4, 2).contiguous().view(-1, 4)
            is_useful = rpn_labels[lvl] >= 0
            
            if is_useful.long().sum().item() == 0:
                rpn_classif_loss = 0
            else:
                rpn_classif_loss = F.binary_cross_entropy_with_logits(rpn_score[is_useful], rpn_labels[lvl][is_useful].float(), size_average=False) / N / 256
            rpn_loss_box_reg = smooth_l1_loss(rpn_box_delta, rpn_box_targets[lvl], rpn_bbox_inside_weights[lvl], rpn_bbox_outside_weights[lvl]) / N
            rpn_classif_losses.append(rpn_classif_loss)
            rpn_losses_box_reg.append(rpn_loss_box_reg)
        rpn_classif_loss = sum(rpn_classif_losses)
        rpn_loss_box_reg = sum(rpn_losses_box_reg)

        if False:  # args.rpn_only:
            loss = rpn_classif_loss + rpn_loss_box_reg
            losses.update(loss.item(), is_useful.float().sum().item())
        else:
            loss = loss_classif + loss_box_reg + rpn_classif_loss + rpn_loss_box_reg

        loss = dict(classification_loss=loss_classif, localization_loss=loss_box_reg, rpn_classification_loss=rpn_classif_loss, rpn_localization_loss=rpn_loss_box_reg)
        return loss


    def forward(self, images, targets=None):
        device = images.device

        """
        imgs_size = torch.tensor([bbox.size[::-1] for bbox in targets], dtype=torch.float32, device=device)

        gt_boxes, gt_labels, img_idx_gt = (t.to(device) for t in join_targets(targets))
        """
        gt_boxes, gt_labels, img_idx_gt = None, None, None
        imgs_size = targets

        features = self._backbone(images)
        rpn_scores, rpn_box_deltas = self.rpn(features)

        all_anchors = []
        img_idx_proposal = []
        inds_inside = []
        for lvl, rpn_score in enumerate(rpn_scores):
            map_shape = rpn_score.shape[-2:]
            a, i, ii = self.rpn._generate_anchors(map_shape, imgs_size, lvl)
            all_anchors.append(a)
            img_idx_proposal.append(i)
            inds_inside.append(ii)
        """
        if self.rpn_only:
            if gt_boxes is not None:
                rpn_labels, rpn_box_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = self.rpn.prepare_targets(all_anchors, gt_boxes, img_idx_proposal, img_idx_gt, inds_inside)
                return rpn_scores, rpn_box_deltas, rpn_labels, rpn_box_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
            return rpn_scores, rpn_box_deltas, all_anchors, img_idx_proposal, inds_inside
        """

        with torch.no_grad():
            rpn_logits = [F.sigmoid(rpn_score) for rpn_score in rpn_scores]
            boxes, selected_scores, idxs = self.rpn.select_top(all_anchors, rpn_logits, rpn_box_deltas, imgs_size)

            if True:
                rois = torch.cat(boxes, 0)
                new_scores = torch.cat(selected_scores, 0)
                new_idxs = torch.cat(idxs, 0)

                post_nms_top_n = 2000
                post_nms_top_n = min(post_nms_top_n, new_scores.shape[0])
                _, new_inds_sorted = torch.topk(new_scores, post_nms_top_n, dim=0, sorted=True)
                rois = rois[new_inds_sorted]
                new_idxs = new_idxs[new_inds_sorted]
                
                lvl_min = 2
                lvl_max = 5
                lvls = map_rois_to_fpn_levels(rois, lvl_min, lvl_max)

                new_boxes = []
                new_idxs_ = []
                rois_idx_order = []
                for lvl in range(lvl_min, lvl_max + 1):
                    idx_lvl = torch.nonzero(lvls == lvl)
                    idx_lvl = idx_lvl.squeeze(1) if idx_lvl.numel() else idx_lvl
                    # print(idx_lvl.shape, lvl)
                    new_boxes.append(rois[idx_lvl])
                    new_idxs_.append(new_idxs[idx_lvl])
                    rois_idx_order.append(idx_lvl)
                _, rois_idx_restore = torch.sort(torch.cat(rois_idx_order, 0))

                # just to keep the other dimension
                new_boxes.append(rois.new())
                # new_idxs_.append(rois_idx_restore.new())
                new_idxs_.append(torch.empty(0, dtype=torch.int64, device=device))

                boxes = new_boxes
                idxs = new_idxs_
                # from IPython import embed; embed()

        if gt_boxes is not None:
            rpn_labels, rpn_box_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = self.rpn.prepare_targets(all_anchors, gt_boxes, img_idx_proposal, img_idx_gt, inds_inside)
            labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, idxs_, num_pos  = self.prepare_targets(boxes, gt_boxes, idxs, img_idx_gt, gt_labels)

        pooled_imgs = []
        # drop P6 for the detector
        for c, (feature, b, i, sc) in enumerate(zip(features[:-1], boxes[:-1], idxs[:-1], (1. / 4, 1. / 8, 1. / 16, 1. / 32))):
            # print(idxs_[c].numel())
            if i.numel() == 0:
                pooled_imgs.append(feature.new())
                continue
            rois = torch.cat([i[:, None].float(), b], dim=1)
            
            if gt_boxes is not None:
                rois = rois[idxs_[c]]

            pooled_img = roi_align(feature, rois, (7, 7), sc, 2)
            pooled_imgs.append(pooled_img)

        pooled_imgs = torch.cat(pooled_imgs, dim=0)
        # pooled_imgs = pooled_imgs[rois_idx_restore]

        scores, box_deltas = self._classifier(pooled_imgs)

        # boxes = torch.cat(boxes[:-1], dim=0)[rois_idx_restore]
        boxes = torch.cat(boxes[:-1], dim=0)

        if gt_boxes is not None:
            return scores, box_deltas, rpn_scores, rpn_box_deltas, rpn_labels, rpn_box_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, num_pos

        return scores, box_deltas, rpn_scores, rpn_box_deltas, boxes, all_anchors, img_idx_proposal, inds_inside, idxs
        # return scores, box_deltas, rpn_scores, rpn_box_deltas, boxes, all_anchors, img_idx_proposal, inds_inside, idxs, bb, pp

    def prepare_targets(self, proposals, gt_boxes, img_idx_proposal, img_idx_gt, gt_labels):

        # TODO maybe make this a list of matrices if this is too big
        shapes = [i.numel() for i in img_idx_proposal]
        device = img_idx_proposal[0].device
        fpn_idxs = torch.tensor([i for i, x in enumerate(img_idx_proposal) for _ in range(x.numel())], device=device)
        proposals = torch.cat(proposals, dim=0)
        img_idx_proposal = torch.cat(img_idx_proposal, dim=0)
        ## from IPython import embed; embed()

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
            expanded_box_targets = torch.zeros(num_proposals * num_classes, 4, dtype=box_targets.dtype, device=device)
            if num_pos > 0:
                map_box_idxs = labels[:num_pos] + torch.arange(num_pos, device=device) * num_classes  # TODO add step instead of multiply in arange
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

        f_labels = []
        f_box_targets = []
        f_i_weights = []
        f_o_weights = []
        f_sampled = []
        n_fpn_idxs = fpn_idxs[sampled_idxs]
        start = 0
        rois_idx_order = []
        for lvl, ss in enumerate(shapes):
            idx_lvl = torch.nonzero(n_fpn_idxs == lvl)
            idx_lvl = idx_lvl.squeeze(1) if idx_lvl.numel() else idx_lvl
            rois_idx_order.append(idx_lvl)
        rois_idx_order = torch.cat(rois_idx_order, 0)
        _, rois_idx_restore = torch.sort(rois_idx_order)

        labels = labels[rois_idx_order]
        expanded_box_targets = expanded_box_targets[rois_idx_order]
        bbox_inside_weights = bbox_inside_weights[rois_idx_order]
        bbox_outside_weights = bbox_outside_weights[rois_idx_order]

        for s, ss in enumerate(shapes):
            end = start + ss
            ii = n_fpn_idxs == s
            # f_labels.append(labels[ii])
            # f_box_targets.append(expanded_box_targets[ii])
            # f_i_weights.append(bbox_inside_weights[ii])
            # f_o_weights.append(bbox_outside_weights[ii])
            f_sampled.append(sampled_idxs[ii] - start)
            assert (f_sampled[-1] >= 0).all()
            start = end

        # return labels, expanded_box_targets, bbox_inside_weights, bbox_outside_weights, sampled_idxs, num_pos
        # return f_labels, f_box_targets, f_i_weights, f_o_weights, f_sampled, num_pos
        return labels, expanded_box_targets, bbox_inside_weights, bbox_outside_weights, f_sampled, num_pos




class RPN_FPN(nn.Module):
    def __init__(self, inplanes, batch_size):
        super(RPN_FPN, self).__init__()
        # anchor_grid_generator = AnchorGridGenerator(16, (32, 64, 128, 256, 512), (0.5, 1, 2), max_size=1333)
        templates = []
        num_anchors = 0
        for lvl in range(2, 7):
            stride = 2 ** lvl
            k_min = 2
            size = 32 * 2 ** (lvl - k_min)
            gen = AnchorGridGenerator(stride, (size,), (0.5, 1, 2), max_size=1500)  # TODO this needs to be bigger than the others, need to fix
            template = gen.generate()
            if torch.cuda.is_available():
                template = template.cuda()
            templates.append(template)
            num_anchors += gen.cell_anchors.shape[0]
        num_anchors = 3

        self.proposal_matcher = Matcher(0.7, 0.3, force_match_for_each_row=True)
        self.fg_bg_sampler = ImageBalancedPositiveNegativeSampler(batch_size, 0.5, which_negatives=-1)
        self.box_coder = BoxCoder(weights=(1., 1., 1., 1.))

        self.anchors_templates = templates

        self.selector = RPNBoxSelector(1000 * 1, 1000 * 1, 0.7, 0)

        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(inplanes, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(inplanes, num_anchors * 4, kernel_size=1, stride=1)

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    # TODO this is per level, but accept batches. confusing
    def _generate_anchors(self, map_shape, im_shape, lvl):
        batch_size = im_shape.size(0)
        device = im_shape.device
        all_anchors = self.anchors_templates[lvl][None]
        height, width = map_shape
        all_anchors = all_anchors[:, :height, :width]
        _, height, width, num_anchors, _ = all_anchors.shape
        all_anchors = all_anchors.expand(batch_size, *all_anchors.shape[1:]).to(device)
        img_idx_proposal_full = torch.arange(batch_size, device=device)[:, None, None, None].expand(*all_anchors.shape[:-1])# .contiguous().view(-1)

        # if im_shape.is_cuda:
        #     all_anchors = all_anchors.cuda()
        #     img_idx_proposal_full = img_idx_proposal_full.cuda()
        # from IPython import embed; embed()

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

    # TODO just use RPN applied N times
    def forward(self, x):
        rpn_scores = []
        bbox_regs = []
        for l in x:
            t = F.relu(self.conv(l))
            logits = self.cls_logits(t)
            bbox_reg = self.bbox_pred(t)
            rpn_scores.append(logits)
            bbox_regs.append(bbox_reg)
        return rpn_scores, bbox_regs

    def prepare_targets(self, proposals, gt_boxes, img_idx_proposal, img_idx_gt, proposals_ids_inside):

        # TODO replace by reshape when merged
        # TODO can maybe remove the permute -- the order is maybe not necessary
        total_anchors = sum(i.numel() for i in img_idx_proposal)  # TODO improve, looks ugly
        shapes = [i.numel() for i in img_idx_proposal]
        batch_size = proposals[0].shape[0]
        # proposals = proposals.permute(0, 3, 1, 2, 4).contiguous().view(-1, 4)
        # img_idx_proposal = img_idx_proposal.permute(0, 3, 1, 2).contiguous().view(-1)
        # proposals_ids_inside = proposals_ids_inside.permute(0, 3, 1, 2).contiguous().view(-1)

        proposals = torch.cat([p.permute(0, 3, 1, 2, 4).reshape(-1, 4) for p in proposals], dim=0)
        img_idx_proposal = torch.cat([p.permute(0, 3, 1, 2).reshape(-1) for p in img_idx_proposal], dim=0)
        proposals_ids_inside = torch.cat([p.permute(0, 3, 1, 2).reshape(-1) for p in proposals_ids_inside], dim=0)


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
        bbox_outside_weights[sampled_idxs] = 1.0 / sampled_idxs.shape[0] * batch_size  ## TODO looks like they divide by the number of ROIs PER IMAGE, not global THIS IS AN APPROXIMATION!

        # unmap to original ids
        exp_labels = labels.new(total_anchors).fill_(-1)
        exp_labels[proposals_ids_inside] = labels
        exp_box_targets = expanded_box_targets.new(total_anchors, 4).fill_(0)
        exp_box_targets[proposals_ids_inside] = expanded_box_targets
        exp_bbox_inside_weights = bbox_inside_weights.new(total_anchors, 4).fill_(0)
        exp_bbox_inside_weights[proposals_ids_inside] = bbox_inside_weights
        exp_bbox_outside_weights = bbox_outside_weights.new(total_anchors, 4).fill_(0)
        exp_bbox_outside_weights[proposals_ids_inside] = bbox_outside_weights

        f_labels = []
        f_box_targets = []
        f_i_weights = []
        f_o_weights = []
        start = 0
        for s in shapes:
            end = start + s
            f_labels.append(exp_labels[start:end])
            f_box_targets.append(exp_box_targets[start:end])
            f_i_weights.append(exp_bbox_inside_weights[start:end])
            f_o_weights.append(exp_bbox_outside_weights[start:end])
            start = end

        # return labels, expanded_box_targets, bbox_inside_weights, bbox_outside_weights
        # return exp_labels, exp_box_targets, exp_bbox_inside_weights, exp_bbox_outside_weights
        return f_labels, f_box_targets, f_i_weights, f_o_weights



    def select_top(self, anchors, rpn_scores, rpn_box_deltas, im_shapes):
        all_boxes = []
        all_scores = []
        all_idxs = []
        for anchor, scores, deltas in zip(anchors, rpn_scores, rpn_box_deltas):
            boxes, s, idxs = self.selector.select_top(anchor, scores, deltas, im_shapes)
            all_boxes.append(boxes)
            all_scores.append(s)
            all_idxs.append(idxs)
        return all_boxes, all_scores, all_idxs


from torchvision.layers import nms
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
        device = rpn_scores.device

        # TODO replace by reshape when merged
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(N, -1)
        rpn_box_deltas = rpn_box_deltas.view(N, A, 4, H, W).permute(0, 3, 4, 1, 2).contiguous().view(N, -1, 4)

        pre_nms_top_n = min(self.pre_nms_top_n, rpn_scores.shape[1])
        rpn_scores, order = rpn_scores.topk(pre_nms_top_n, dim=1, sorted=True)

        # TODO attention with the GPU device
        order = (torch.arange(N, device=device)[:, None], order)
        rpn_box_deltas = rpn_box_deltas[order]
        anchors = anchors.contiguous().view(N, -1, 4)[order]

        assert rpn_box_deltas.numel() > 0
        assert anchors.numel() > 0

        proposals = self.box_coder.decode(rpn_box_deltas.view(-1, 4), anchors.view(-1, 4))

        proposals = proposals.view(N, -1, 4)

        # TODO optimize / make batch friendly
        sampled_proposals = []
        sampled_scores = []
        for proposal, score, im_shape in zip(proposals, rpn_scores, im_shapes):
            height, width = im_shape
            
            if proposal.dim() == 0:
                sampled_proposals.append(proposal.new())
                sampled_scores.append(score.new())
                print('skipping')
                continue
            
            p = box_utils.clip_boxes_to_image(proposal, height, width)
            keep = _filter_boxes(p, self.min_size, im_shape)
            p = p[keep, :]
            score = score[keep]
            if self.nms_thresh > 0:
                # keep = box_utils.nms(p, score, self.nms_thresh)
                # keep = torch.from_numpy(cython_nms(torch.cat((p, score[:, None]), 1).detach().cpu().numpy(), self.nms_thresh)).cuda()
                keep = nms(p.cpu(), score.cpu(), self.nms_thresh).to(device)
                if self.post_nms_top_n > 0:
                    keep = keep[:self.post_nms_top_n]
                p = p[keep]
                score = score[keep]
            sampled_proposals.append(p)
            sampled_scores.append(score)

        ids = [i for i, x in enumerate(sampled_proposals) for _ in range(len(x))]
        ids = torch.tensor(ids, device=device)
        sampled_proposals = torch.cat(sampled_proposals, dim=0)

        if True:
            sampled_scores = torch.cat(sampled_scores, dim=0)

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
        (x_ctr < im_shape[1].item()) & (y_ctr < im_shape[0].item())).squeeze(1)
    return keep



from .models_caffe2.rpn_resnet50 import ResNet50_FPN
class C2ResNetFPN(FPN, ResNet50_FPN):
    def __init__(self, weights=None, num_classes=81, rpn_batch_size=None, batch_size=None, rpn_only=None):
        super(C2ResNetFPN, self).__init__(num_classes=num_classes, rpn_batch_size=batch_size, batch_size=batch_size, rpn_only=rpn_only)
        if weights is not None:
            import pickle
            if isinstance(weights, str):  # TODO py2 compat
                with open(weights, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                weights = data['blobs']
            weights = self._parse_weights(weights)
            self.load_state_dict(weights, strict=False)



def map_rois_to_fpn_levels(rois, k_min, k_max):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """
    # Compute level ids
    s = torch.sqrt(boxes_area(rois))
    s0 = 224  # cfg.FPN.ROI_CANONICAL_SCALE  # default: 224
    lvl0 = 4  # cfg.FPN.ROI_CANONICAL_LEVEL  # default: 4

    # Eqn.(1) in FPN paper
    target_lvls = torch.floor(lvl0 + torch.log2(s / s0 + 1e-6))
    target_lvls = torch.clamp(target_lvls, min=k_min, max=k_max)
    return target_lvls


if __name__ == "__main__":
    m = C2ResNetFPN('/private/home/fmassa/github/detectron.pytorch/torch_detectron/fpn-r50.pkl')
    from IPython import embed; embed()
