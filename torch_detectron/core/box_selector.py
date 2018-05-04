import torch
from torchvision.layers import nms

from .box_coder import BoxCoder

from torchvision.structures.bounding_box import BBox


#TODO add option for different params in train / test
class RPNBoxSelector(object):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the proposals
    to the heads
    """
    def __init__(self, pre_nms_top_n, post_nms_top_n, nms_thresh, min_size,
            box_coder=BoxCoder(weights=(1., 1., 1., 1.))):
        # TODO ATTENTION, as those numbers are for single-image in Detectron, and here it's for the batch
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size

        self.box_coder = box_coder

    def forward_for_single_feature_map(self, anchors, objectness, box_regression):
        """
        anchors: list of BBox
        objectness: tensor of size N, A, H, W
        box_regression: tensor of size N, A * 4, H, W
        """
        device = objectness.device
        N, A, H, W = objectness.shape

        # put in the same format as anchors
        objectness = objectness.permute(0, 2, 3, 1).reshape(N, -1)
        objectness = objectness.sigmoid()
        box_regression = box_regression.view(N, -1, 4, H, W).permute(0, 3, 4, 1, 2)
        box_regression = box_regression.reshape(N, -1, 4)

        num_anchors = A * H * W

        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)

        # TODO check if this batch_idx is really needed
        batch_idx = torch.arange(N, device=device)[:, None]
        box_regression = box_regression[batch_idx, topk_idx]

        image_shapes = [box.size[::-1] for box in anchors]
        concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)
        concat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]

        proposals = self.box_coder.decode(
                box_regression.view(-1, 4), concat_anchors.view(-1, 4))

        proposals = proposals.view(N, -1, 4)

        # TODO optimize / make batch friendly
        sampled_bboxes = []
        for proposal, score, im_shape in zip(proposals, objectness, image_shapes):
            height, width = im_shape
            
            if proposal.dim() == 0:
                sampled_proposals.append(proposal.new())
                sampled_scores.append(score.new())
                print('skipping')
                continue
            
            p = _clip_boxes_to_image(proposal, height, width)
            keep = _filter_boxes(p, self.min_size, im_shape)
            p = p[keep, :]
            score = score[keep]
            if self.nms_thresh > 0:
                keep = nms(p.cpu(), score.cpu(), self.nms_thresh)
                if self.post_nms_top_n > 0:
                    keep = keep[:self.post_nms_top_n]
                p = p[keep]
                score = score[keep]
            sampled_bbox = BBox(p, (width, height), mode='xyxy')
            sampled_bbox.add_field('objectness', score)
            sampled_bboxes.append(sampled_bbox)
            # TODO maybe also copy the other fields that were originally present?

        return sampled_bboxes

    def __call__(self, anchors, objectness, box_regression):
        assert len(anchors) == 1, 'only single feature map supported'
        sampled_boxes = []
        for a, o, b in zip(anchors, objectness, box_regression):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))

        return sampled_boxes


class FPNRPNBoxSelector(RPNBoxSelector):
    def __init__(self, roi_to_fpn_level_mapper, fpn_post_nms_top_n, **kwargs):
        super(FPNRPNBoxSelector, self).__init__(**kwargs)
        self.roi_to_fpn_level_mapper = roi_to_fpn_level_mapper
        self.fpn_post_nms_top_n = fpn_post_nms_top_n

    def __call__(self, anchors, objectness, box_regression):
        sampled_boxes = []
        for a, o, b in zip(anchors, objectness, box_regression):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))

        # shortcut for single feature maps
        if len(sampled_boxes) == 1:
            return sampled_boxes

        # merge all lists
        num_features = len(sampled_boxes)
        num_images = len(sampled_boxes[0])

        merged_lists = [box for per_feature_boxes in sampled_boxes for box in per_feature_boxes]
        image_sizes = [b.size for b in sampled_boxes[0]]

        device = merged_lists[0].bbox.device
        indices = [torch.full((box.bbox.shape[0],), img_idx, device=device) for per_feature_boxes in sampled_boxes
                for img_idx, box in enumerate(per_feature_boxes)]

        # TODO make it a helper function
        concat_boxes = torch.cat([b.bbox for b in merged_lists], dim=0)
        indices = torch.cat(indices, dim=0)
        extra_fields = {}
        field_names = merged_lists[0].fields()
        for field in field_names:
            extra_fields[field] = torch.cat([b.get_field(field) for b in merged_lists], dim=0)

        post_nms_top_n = min(self.fpn_post_nms_top_n, concat_boxes.shape[0])
        _, inds_sorted = torch.topk(extra_fields['objectness'], post_nms_top_n, dim=0, sorted=True)

        concat_boxes = concat_boxes[inds_sorted]
        indices = indices[inds_sorted]
        for field, data in extra_fields.items():
            extra_fields[field] = data[inds_sorted]

        levels = self.roi_to_fpn_level_mapper(concat_boxes)

        # maps back to the original order
        boxes = []
        lvl_min = self.roi_to_fpn_level_mapper.k_min
        lvl_max = self.roi_to_fpn_level_mapper.k_max
        for feat_lvl in range(lvl_min, lvl_max + 1):
            per_feat_boxes = []
            for img_idx in range(num_images):
                lvl_idx_per_img = (indices == img_idx) & (levels == feat_lvl)
                selected_boxes = concat_boxes[lvl_idx_per_img]
                bbox = BBox(selected_boxes, image_sizes[img_idx], mode='xyxy')
                for field, data in extra_fields.items():
                    bbox.add_field(field, data[lvl_idx_per_img])
                per_feat_boxes.append(bbox)
            boxes.append(per_feat_boxes)

        return boxes


# TODO move this to bounding box class?
def _clip_boxes_to_image(boxes, height, width):
    fact = 1  # TODO REMOVE
    num_boxes = boxes.shape[0]
    b1 = boxes[:, 0::2].clamp(min=0, max=width-fact)
    b2 = boxes[:, 1::2].clamp(min=0, max=height-fact)
    boxes = torch.stack((b1, b2), 2).view(num_boxes, -1)
    return boxes


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


from .box_ops import boxes_area
class ROI2FPNLevelsMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """
    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, rois):
        # Compute level ids
        s = torch.sqrt(boxes_area(rois))

        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + 1e-6))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls


if __name__ == '__main__':
    from torchvision.structures.bounding_box import BBox
    box_selector = RPNBoxSelector(10, 5, 0.7, 0)

    anchors = [BBox(torch.rand(16, 4) * 50, (100, 100), mode='xywh').convert('xyxy')]
    objectness = torch.rand(1, 4, 2, 2)
    box_regression = torch.rand(1, 4 * 4, 2, 2)

    sampled = box_selector(anchors, objectness, box_regression)

    from IPython import embed; embed()

