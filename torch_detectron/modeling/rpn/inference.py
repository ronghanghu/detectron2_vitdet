import torch

from torch_detectron.layers import nms as box_nms
from torch_detectron.structures.bounding_box import BoxList
from ..box_coder import BoxCoder
from ..utils import cat
from ..utils import cat_bbox
from ..utils import nonzero


# TODO add option for different params in train / test
class RPNBoxSelector(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(
        self, pre_nms_top_n, post_nms_top_n, nms_thresh, min_size, box_coder=None
    ):
        """
        Arguments:
            pre_nms_top_n (int)
            post_nms_top_n (int)
            nms_thresh (float)
            min_size (int)
            box_coder (BoxCoder)
        """
        super(RPNBoxSelector, self).__init__()
        # TODO ATTENTION, as those numbers are for single-image in Detectron,
        # and here it's for the batch
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size

        self.box_coder = (
            BoxCoder(weights=(1.0, 1.0, 1.0, 1.0)) if not box_coder else box_coder
        )

    def add_gt_proposals(self, proposals, targets):
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
            gt_box.add_field(
                "objectness", torch.ones(gt_box.bbox.shape[0], device=device)
            )

        proposals = [
            cat_bbox((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def forward_for_single_feature_map(self, anchors, objectness, box_regression):
        """
        Arguments:
            anchors: list[BoxList]
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

        batch_idx = torch.arange(N, device=device)[:, None]
        box_regression = box_regression[batch_idx, topk_idx]

        image_shapes = [box.size[::-1] for box in anchors]
        concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)
        concat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]

        proposals = self.box_coder.decode(
            box_regression.view(-1, 4), concat_anchors.view(-1, 4)
        )

        proposals = proposals.view(N, -1, 4)

        # TODO optimize / make batch friendly
        sampled_bboxes = []
        for proposal, score, im_shape in zip(proposals, objectness, image_shapes):
            height, width = im_shape

            p = _clip_boxes_to_image(proposal, height, width)
            keep = _filter_boxes(p, self.min_size, im_shape)
            p = p[keep, :]
            score = score[keep]
            if self.nms_thresh > 0:
                keep = box_nms(p, score, self.nms_thresh)
                if self.post_nms_top_n > 0:
                    keep = keep[: self.post_nms_top_n]
                p = p[keep]
                score = score[keep]
            sampled_bbox = BoxList(p, (width, height), mode="xyxy")
            sampled_bbox.add_field("objectness", score)
            sampled_bboxes.append(sampled_bbox)
            # TODO maybe also copy the other fields that were originally present?

        return sampled_bboxes

    def forward(self, anchors, objectness, box_regression, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            objectness: list[tensor]
            box_regression: list[tensor]
        """
        num_levels = len(objectness)
        assert num_levels == 1, "only single feature map supported"
        anchors = list(zip(*anchors))
        sampled_boxes = []
        for a, o, b in zip(anchors, objectness, box_regression):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))

        # there is a single feature map, remove the extra nesting level
        sampled_boxes = sampled_boxes[0]

        # append ground-truth bboxes to proposals
        if self.training and targets is not None:
            sampled_boxes = self.add_gt_proposals(sampled_boxes, targets=targets)

        return sampled_boxes


class FPNRPNBoxSelector(RPNBoxSelector):
    def __init__(self, roi_to_fpn_level_mapper, fpn_post_nms_top_n, **kwargs):
        """
        Arguments:
            roi_to_fpn_level_mapper (ROI2FPNLevelMapper)
            fpn_post_nms_top_n (int)
            + same arguments as RPNBoxSelector
        """
        super(FPNRPNBoxSelector, self).__init__(**kwargs)
        self.roi_to_fpn_level_mapper = roi_to_fpn_level_mapper
        self.fpn_post_nms_top_n = fpn_post_nms_top_n

    def forward(self, anchors, objectness, box_regression, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            objectness: list[tensor]
            box_regression: list[tensor]
        """
        sampled_boxes = []
        num_levels = len(objectness)
        anchors = list(zip(*anchors))
        for a, o, b in zip(anchors, objectness, box_regression):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_bbox(boxlist) for boxlist in boxlists]

        num_images = len(boxlists)
        # different behavior during training and during testing:
        # during training, post_nms_top_n is over *all* the proposals combined, while
        # during testing, it is over the proposals for each image
        # TODO resolve this difference and make it consistent. It should be per image,
        # and not per batch
        if self.training:
            objectness = torch.cat([boxlist.get_field("objectness") for boxlist in boxlists], dim=0)
            box_sizes = [boxlist.bbox.shape[0] for boxlist in boxlists]
            post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
            inds_mask = torch.zeros_like(objectness, dtype=torch.uint8)
            inds_mask[inds_sorted] = 1
            inds_mask = inds_mask.split(box_sizes)
            for i in range(num_images):
                boxlists[i] = boxlists[i][inds_mask[i]]

        else:
            for i in range(num_images):
                objectness = boxlists[i].get_field("objectness")
                post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
                _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
                boxlists[i] = boxlists[i][inds_sorted]

        """
        # TODO almost all this part can be
        # factored out in a RPN-agnostic class
        # merge all lists
        num_images = len(sampled_boxes[0])

        merged_lists = [
            box for per_feature_boxes in sampled_boxes for box in per_feature_boxes
        ]
        image_sizes = [b.size for b in sampled_boxes[0]]

        device = merged_lists[0].bbox.device
        indices = [
            torch.full((box.bbox.shape[0],), img_idx, device=device)
            for per_feature_boxes in sampled_boxes
            for img_idx, box in enumerate(per_feature_boxes)
        ]

        # TODO make these concatenations a helper function?
        # once we make train and test consistent, this can be greatly
        # simplified, as it will go per image, and not per batch
        concat_boxes = torch.cat([b.bbox for b in merged_lists], dim=0)
        indices = torch.cat(indices, dim=0)
        extra_fields = {}
        field_names = merged_lists[0].fields()
        for field in field_names:
            extra_fields[field] = torch.cat(
                [b.get_field(field) for b in merged_lists], dim=0
            )

        post_nms_top_n = min(self.fpn_post_nms_top_n, concat_boxes.shape[0])
        # different behavior during training and during testing:
        # during training, post_nms_top_n is over *all* the proposals combined, while
        # during testing, it is over the proposals for each image
        # TODO resolve this difference and make it consistent. It should be per image,
        # and not per batch
        if self.training:
            _, inds_sorted = torch.topk(
                extra_fields["objectness"], post_nms_top_n, dim=0, sorted=True
            )
        else:
            inds_sorted = []
            for i in range(num_images):
                objectness = extra_fields["objectness"].clone()
                objectness[indices != i] = -1
                _, inds_sorted_img = torch.topk(
                    objectness, post_nms_top_n, dim=0, sorted=True
                )
                inds_sorted.append(inds_sorted_img)
            inds_sorted = cat(inds_sorted, dim=0)

        concat_boxes = concat_boxes[inds_sorted]
        indices = indices[inds_sorted]
        for field, data in extra_fields.items():
            extra_fields[field] = data[inds_sorted]

        boxlists = []
        # convert back to BoxList
        for img_idx in range(num_images):
            idx_per_img = nonzero(indices == img_idx)[0]
            boxes_per_img = concat_boxes[idx_per_img]
            boxlist_per_img = BoxList(boxes_per_img, image_sizes[img_idx], mode="xyxy")
            for field, data in extra_fields.items():
                boxlist_per_img.add_field(field, data[idx_per_img])
            boxlists.append(boxlist_per_img)
        """
        # append ground-truth bboxes to proposals
        if self.training and targets is not None:
            boxlists = self.add_gt_proposals(boxlists, targets)

        return boxlists


# TODO move this to bounding box class?
def _clip_boxes_to_image(boxes, height, width):
    fact = 1  # TODO REMOVE
    num_boxes = boxes.shape[0]
    b1 = boxes[:, 0::2].clamp(min=0, max=width - fact)
    b2 = boxes[:, 1::2].clamp(min=0, max=height - fact)
    boxes = torch.stack((b1, b2), 2).view(num_boxes, -1)
    return boxes


def _filter_boxes(boxes, min_size, im_shape):
    """Only keep boxes with both sides >= min_size and center within the image.
    """
    # Scale min_size to match image scale
    fact = 1  # TODO remove
    ws = boxes[:, 2] - boxes[:, 0] + fact
    hs = boxes[:, 3] - boxes[:, 1] + fact
    x_ctr = boxes[:, 0] + ws / 2.0
    y_ctr = boxes[:, 1] + hs / 2.0
    keep = nonzero(
        (ws >= min_size)
        & (hs >= min_size)
        & (x_ctr < im_shape[1])
        & (y_ctr < im_shape[0])
    )[0]
    return keep
