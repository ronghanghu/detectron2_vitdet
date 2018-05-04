import torch
from torch.autograd import Variable
from torch.nn import functional as F

from lib.utils.box import box_iou



BELOW_UNMATCHED_THRESHOLD = -1
BETWEEN_THRESHOLDS = -2

class Matcher(object):
    def __init__(self, matched_threshold, unmatched_threshold, force_match_for_each_row=False):
        self.matched_threshold = matched_threshold
        self.unmatched_threshold = unmatched_threshold
        self.force_match_for_each_row = force_match_for_each_row

    def __call__(self, match_quality_matrix):
        matched_vals, matches = match_quality_matrix.max(0)
        below_unmatched_threshold = matched_vals < self.unmatched_threshold
        between_thresholds = ((matched_vals >= self.unmatched_threshold)
                & (matched_vals < self.matched_threshold))

        # Ross implementation always uses the box with the max overlap
        matched_backup = matches.clone()
        # TODO this is the convention of TF
        matches[below_unmatched_threshold] = BELOW_UNMATCHED_THRESHOLD
        matches[between_thresholds] = BETWEEN_THRESHOLDS

        if self.force_match_for_each_row:
            force_matched_vals, force_matches = match_quality_matrix.max(1)
            force_max_overlaps = torch.nonzero(match_quality_matrix == force_matched_vals[:, None])
            # matches[force_max_overlaps[:, 1]] = force_max_overlaps[:, 0]
            # Ross implementation always uses the box with the max overlap
            matches[force_max_overlaps[:, 1]] = matched_backup[force_max_overlaps[:, 1]]
        return matches

class IoUSimilarity(object):
    def __call__(self, boxlist1, boxlist2):
        return box_iou(boxlist1, boxlist2)

def batch_box_iou(boxlist1, boxlist2, boxlist1_ids, boxlist2_ids):
    # general approach: instead of a for loop, compute a large
    # matching matrix and zero the entries that do not match
    # TODO annoying, need to wrap/unwrap in variables
    iou = box_iou(boxlist1, boxlist2)
    # mask correspondences that do not belong to the same image
    # TODO attention if that is a too large matrix
    iou[boxlist1_ids[:, None] != boxlist2_ids[None]] = -1
    return iou

class TargetAssigner(object):
    def __init__(self, similarity_calc, matcher, box_coder,
            negative_class_weight=1.0, unmatched_cls_target=None):
        pass

    def assign(self, anchors, groundtruth_boxes, groundtruth_labels=None,
               groundtruth_weights=None, **params):
        similirity_matrix = self.similarity_calc(groundtruth_boxes, anchors)

# TODO this class samples uniformly over the batch, and not over each image
class BalancedPositiveNegativeSampler(object):
    def __init__(self, batch_size, positive_fraction):
        self.batch_size = batch_size
        self.positive_fraction = positive_fraction

    def __call__(self, indices):
        positive = (indices >= 0).nonzero()
        if positive.numel() > 0:
            positive = positive.squeeze(1)
        negative = (indices == BETWEEN_THRESHOLDS).nonzero()
        if negative.numel() > 0:
            negative = negative.squeeze(1)

        num_pos = int(self.batch_size * self.positive_fraction)
        num_pos = min(positive.size(0), num_pos) if positive.numel() else 0
        # TODO RPN sample negatives with replacement, but not Fast R-CNN
        num_neg = self.batch_size - num_pos
        num_neg = min(negative.size(0), num_neg) if negative.numel() else 0

        if positive.numel() > 0:
            perm1 = torch.randperm(positive.size(0))
            if positive.is_cuda:
                with torch.cuda.device_of(positive):
                    perm1 = perm1.cuda()
            pos_idx = positive[perm1[:num_pos]]
        else:
            pos_idx = positive.new().long()
        if negative.numel() > 0:
            perm2 = torch.randperm(negative.size(0))
            if negative.is_cuda:
                with torch.cuda.device_of(negative):
                    perm2 = perm2.cuda()
            neg_idx = negative[perm2[:num_neg]]
        else:
            neg_idx = negative.new().long()

        # TODO maybe merge both in the same array
        # return pos_idx, neg_idx
        return torch.cat([pos_idx, neg_idx], 0), pos_idx.size(0) if pos_idx.numel() else 0

# TODO might be implemented more efficiently using torch.multinomial
class ImageBalancedPositiveNegativeSampler(object):
    def __init__(self, batch_size, positive_fraction, which_negatives=BETWEEN_THRESHOLDS):
        self.batch_size = batch_size
        self.positive_fraction = positive_fraction
        self.which_negatives = which_negatives

    def __call__(self, indices, img_idx):
        # TODO this is hacky, improve!
        num_imgs = img_idx.max().item() + 1
        pos = []
        negs = []
        total_pos = 0
        for i in range(num_imgs):
            img_id_mask = img_idx == i
            positive = ((indices >= 0) & img_id_mask).nonzero()
            if positive.numel() > 0:
                positive = positive.squeeze(1)
            negative = ((indices == self.which_negatives) & img_id_mask).nonzero()
            if negative.numel() > 0:
                negative = negative.squeeze(1)

            num_pos = int(self.batch_size * self.positive_fraction / num_imgs)
            num_pos = min(positive.size(0), num_pos) if positive.numel() else 0
            # TODO RPN sample negatives with replacement, but not Fast R-CNN
            num_neg = self.batch_size // num_imgs - num_pos
            num_neg = min(negative.size(0), num_neg) if negative.numel() else 0

            if positive.numel() > 0:
                perm1 = torch.randperm(positive.size(0))
                if positive.is_cuda:
                    with torch.cuda.device_of(positive):
                        perm1 = perm1.cuda()
                pos_idx = positive[perm1[:num_pos]]
            else:
                pos_idx = positive.new().long()
            if negative.numel() > 0:
                perm2 = torch.randperm(negative.size(0))
                if negative.is_cuda:
                    with torch.cuda.device_of(negative):
                        perm2 = perm2.cuda()
                neg_idx = negative[perm2[:num_neg]]
            else:
                neg_idx = negative.new().long()
            pos.append(pos_idx)
            negs.append(neg_idx)
            total_pos += pos_idx.numel()

        # TODO maybe merge both in the same array
        # return pos_idx, neg_idx
        # return torch.cat([pos_idx, neg_idx], 0), pos_idx.size(0) if pos_idx.numel() else 0
        # print(len(pos[0]), len(pos[1]), len(negs))
        # print(pos, negs)
        pos = torch.cat(pos, 0) if sum(t.numel() for t in pos) > 0 else positive.new().long()
        negs = torch.cat(negs, 0) if sum(t.numel() for t in negs) > 0 else negative.new().long()
        concat = torch.cat([pos, negs], 0) if pos.numel() > 0 or negs.numel() > 0 else negs.new().long()
        # return torch.cat([pos, negs], 0), total_pos
        return concat, total_pos



