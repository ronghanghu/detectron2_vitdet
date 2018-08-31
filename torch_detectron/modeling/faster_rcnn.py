import torch
import torch.nn.functional as F
from torch import nn

from torch_detectron.layers import ROIAlign

from .utils import cat


# FIXME Rename to RPNOutputs
class RPNHeads(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, inplanes, num_anchors):
        """
        Arguments:
            inplanes (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHeads, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(inplanes, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(inplanes, num_anchors * 4, kernel_size=1, stride=1)

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


class Pooler(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BBox.
    """

    def __init__(self, output_size, scales, sampling_ratio, drop_last):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[flaot]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
            drop_last (bool): if passed, drop the last feature map
        """
        super(Pooler, self).__init__()
        poolers = []
        for scale in scales:
            poolers.append(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio
                )
            )
        self.poolers = nn.ModuleList(poolers)
        self.drop_last = drop_last

    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[list[BBox]]): boxes to be used to perform the pooling operation.
                The first level corresponds to the feature map level, and the second
                to different input images.
        Returns:
            result (Tensor)
        """
        if self.drop_last:
            x = x[:-1]
        assert len(boxes) == len(self.poolers)
        assert len(boxes) == len(x)
        result = []
        for per_level_feature, per_level_boxes, pooler in zip(x, boxes, self.poolers):
            # TODO the scales can be inferred from the bboxes and the feature maps
            # print(per_level_boxes[0].size[::-1], per_level_feature.shape)
            ids = [
                i for i, l in enumerate(per_level_boxes) for _ in range(l.bbox.shape[0])
            ]
            concat_boxes = torch.cat([b.bbox for b in per_level_boxes], dim=0)
            ids = concat_boxes.new_tensor(ids)
            if ids.numel() == 0:
                continue
            concat_boxes = torch.cat([ids[:, None], concat_boxes], dim=1)
            result.append(pooler(per_level_feature, concat_boxes))
        if not result:  # empty
            return x[0].new()
        return cat(result, dim=0)
