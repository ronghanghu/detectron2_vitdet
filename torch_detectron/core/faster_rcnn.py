import torch
from torch import nn
import torch.nn.functional as F

from torch_detectron.layers import ROIAlign


class RPNHeads(nn.Module):
    def __init__(self, inplanes, num_anchors):
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
    def __init__(self, pooler=None):
        """
        Arguments:
            pooler (nn.Module)
        """
        super(Pooler, self).__init__()
        if pooler is None:
            # FIXME only for backward-compatibility. Remove this
            pooler = ROIAlign((14, 14), spatial_scale=1.0 / 16, sampling_ratio=0)
        self.pooler = pooler

    def forward(self, x, boxes):
        """
        Arguments:
            x (list of tensor)
            boxes (list of list of BBox)
        """
        result = []
        for per_level_feature, per_level_boxes in zip(x, boxes):
            ids = [i for i, l in enumerate(per_level_boxes) for _ in range(l.bbox.shape[0])]
            concat_boxes = torch.cat([b.bbox for b in per_level_boxes], dim=0)
            ids = concat_boxes.new_tensor(ids)
            concat_boxes = torch.cat([ids[:, None], concat_boxes], dim=1)
            result.append(self.pooler(per_level_feature, concat_boxes))
        return result[0] if len(result) == 1 else torch.cat(result, dim=0)
