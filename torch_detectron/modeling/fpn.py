import torch
import torch.nn.functional as F
from torch import nn

from torch_detectron.layers import ROIAlign


class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(self, layers, representation_size, top_blocks=None):
        """
        Arguments:
            layers (list[int]): number of channels for each feature map that
                will be fed
            representation_size (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, layer_size in enumerate(layers, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)
            inner_block_module = nn.Conv2d(layer_size, representation_size, 1)
            layer_block_module = nn.Conv2d(
                representation_size, representation_size, 3, 1, 1
            )
            for module in [inner_block_module, layer_block_module]:
                nn.init.kaiming_uniform_(module.weight, a=1)
                nn.init.constant_(module.bias, 0)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.

        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            inner_top_down = F.upsample(last_inner, scale_factor=2, mode="nearest")
            inner_lateral = getattr(self, inner_block)(feature)
            # TODO use size instead of scale to make it robust to different sizes
            # inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:], mode='bilinear', align_corners=False)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))

        if self.top_blocks is not None:
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        return tuple(results)


class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class FPNPooler(nn.Module):
    """
    Pooler for Detection with FPN.
    It currently hard-code ROIAlign in the implementation, with sampling_ratio=2,
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
        super(FPNPooler, self).__init__()
        poolers = []
        for scale in scales:
            poolers.append(ROIAlign(output_size, spatial_scale=scale, sampling_ratio=2))
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
        return torch.cat(result, dim=0)


class FPNHeadClassifier(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, num_classes, input_size, representation_size):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(FPNHeadClassifier, self).__init__()
        self.fc6 = nn.Linear(input_size, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

        self.cls_score = nn.Linear(representation_size, num_classes)
        self.bbox_pred = nn.Linear(representation_size, num_classes * 4)

        for l in [self.fc6, self.fc7]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

        for l in [self.cls_score, self.bbox_pred]:
            nn.init.normal_(l.weight, std=0.001)
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas
