import torch
from torch import nn
import torch.nn.functional as F

from torchvision.layers import ROIAlign


class FPN(nn.Module):
    def __init__(self, layers, representation_size):
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, layer_size in enumerate(layers, 1):
            inner_block = 'fpn_inner{}'.format(idx)
            layer_block = 'fpn_layer{}'.format(idx)
            self.add_module(inner_block,
                    nn.Conv2d(layer_size, representation_size, 1))
            self.add_module(layer_block,
                    nn.Conv2d(representation_size, representation_size, 3, 1, 1))
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)

    def forward(self, x):

        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(
                x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]):
            inner_top_down = F.upsample(last_inner, scale_factor=2, mode='nearest')
            inner_lateral = getattr(self, inner_block)(feature)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))

        smallest_resolution = F.max_pool2d(results[-1], 1, 2, 0)
        results.append(smallest_resolution)

        return tuple(results)


def fpn_resnet50_conv5_body(pretrained=None):
    from .resnet_builder import ResNetBackbone, Bottleneck
    body = ResNetBackbone(Bottleneck, layers=[(2, 3, True), (3, 4, True), (4, 6, True), (5, 3, True)])
    fpn = FPN(layers=[256, 512, 1024, 2048], representation_size=256)
    if pretrained:
        state_dict = torch.load(pretrained)
        body.load_state_dict(state_dict, strict=False)
        fpn.load_state_dict(state_dict, strict=False)
    model = nn.Sequential(body, fpn)
    return model


class FPNPooler(nn.Module):
    def __init__(self, output_size, scales, sampling_ratio, drop_last):
        super(FPNPooler, self).__init__()
        poolers = []
        for scale in scales:
            poolers.append(ROIAlign(output_size, spatial_scale=scale, sampling_ratio=2))
        self.poolers = nn.ModuleList(poolers)
        self.drop_last = drop_last

    def forward(self, x, boxes):
        if self.drop_last:
            x = x[:-1]
        assert len(boxes) == len(self.poolers)
        result = []
        for per_level_feature, per_level_boxes, pooler in zip(x, boxes, self.poolers):
            # TODO the scales can be inferred from the bboxes and the feature maps
            # print(per_level_boxes[0].size[::-1], per_level_feature.shape)
            ids = [i for i, l in enumerate(per_level_boxes) for _ in range(l.bbox.shape[0])]
            concat_boxes = torch.cat([b.bbox for b in per_level_boxes], dim=0)
            ids = concat_boxes.new_tensor(ids)
            if ids.numel() == 0:
                continue
            concat_boxes = torch.cat([ids[:, None], concat_boxes], dim=1)
            result.append(pooler(per_level_feature, concat_boxes))
        return torch.cat(result, dim=0)


class FPNHeadClassifier(nn.Module):
    def __init__(self, num_classes, input_size, representation_size):
        super(FPNHeadClassifier, self).__init__()
        self.fc6 = nn.Linear(input_size, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

        self.cls_score = nn.Linear(representation_size, num_classes)
        self.bbox_pred = nn.Linear(representation_size, num_classes * 4)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas

def fpn_classification_head(num_classes, pretrained=None):
    model = FPNHeadClassifier(num_classes, 256 * 7 * 7, 1024)
    if pretrained:
        state_dict = torch.load(pretrained)
        model.load_state_dict(state_dict, strict=False)
    return model
