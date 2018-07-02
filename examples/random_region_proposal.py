import torch
from torch import nn

from torch_detectron.structures.bounding_box import BBox


class RandomRegionProposal(nn.Module):
    """
    This is an example implementation. One would generally use
    algorithms such as selective search or edge-box for
    generating the proposals
    """
    def __init__(self, number_of_proposals):
        super(RandomRegionProposal, self).__init__()
        self.number_of_proposals = number_of_proposals

    def forward(self, x, features=None):
        """
        Apply an arbitrary transformation on the image
        to obtain the bounding boxes.
        x: image_list
        """
        bboxes = []
        for (height, width) in x.image_sizes:
            boxes = torch.rand(self.number_of_proposals, 4, device=x.tensors.device)
            boxes[:, [0, 2]] *= width
            boxes[:, [1, 3]] *= height
            bbox = BBox(boxes, (width, height), mode='xywh').convert('xyxy')
            bbox.bbox[:, 0].clamp_(min=0, max=width)
            bbox.bbox[:, 1].clamp_(min=0, max=height)
            bbox.bbox[:, 2].clamp_(min=0, max=width)
            bbox.bbox[:, 3].clamp_(min=0, max=height)
            bboxes.append(bbox)
        # single level
        return [bboxes]

    def predict(self, x, features=None):
        return self(x, features)


class FPNRandomRegionProposal(RandomRegionProposal):
    def __init__(self, number_of_proposals, roi_to_fpn_level_mapper):
        super(FPNRandomRegionProposal, self).__init__(number_of_proposals)
        self.roi_to_fpn_level_mapper = roi_to_fpn_level_mapper

    def forward(self, x, features=None):
        boxes = super(FPNRandomRegionProposal, self).forward(x, features)
        
        # TODO maybe factor this out in a helper class
        image_sizes = x.image_sizes
        lvl_min = self.roi_to_fpn_level_mapper.k_min
        lvl_max = self.roi_to_fpn_level_mapper.k_max
        fpn_boxes = []

        # for each image, split in different fpn levels
        for img_idx, bboxes in enumerate(boxes[0]):
            height, width = image_sizes[img_idx]
            box_data = bboxes.bbox
            per_img_boxes = []
            levels = self.roi_to_fpn_level_mapper(box_data)
            for feat_lvl in range(lvl_min, lvl_max + 1):
                lvl_idx_per_img = (levels == feat_lvl)
                selected_boxes = box_data[lvl_idx_per_img]
                bbox = BBox(selected_boxes, (width, height), mode='xyxy')
                per_img_boxes.append(bbox)
            fpn_boxes.append(per_img_boxes)

        # invert box representation to be first number of levels, and then
        # number of images
        fpn_boxes = list(zip(*fpn_boxes))
        return fpn_boxes
