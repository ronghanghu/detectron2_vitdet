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


if __name__ == '__main__':

    from .resnet_builder import resnet50_conv4_body, resnet50_conv5_head
    from .anchor_generator import AnchorGenerator
    from .box_selector import RPNBoxSelector

    from . import detection_model

    from collections import OrderedDict
    pretrained_path = '/private/home/fmassa/github/detectron.pytorch/torch_detectron/lib/clean_inference/faster_rcnn_resnet50.pth'
    pretrained_state_dict = torch.load(pretrained_path)
    rpn_state_dict = OrderedDict()
    for k in ['conv.weight', 'conv.bias',
            'cls_logits.weight', 'cls_logits.bias',
            'bbox_pred.weight', 'bbox_pred.bias']:
        tensor = pretrained_state_dict['rpn.' + k]
        rpn_state_dict[k] = tensor

    backbone = resnet50_conv4_body(pretrained_path)

    anchor_generator = AnchorGenerator(scales=(0.125, 0.25, 0.5, 1., 2.), anchor_offset=(8.5, 8.5))

    box_selector = RPNBoxSelector(6000, 1000, 0.7, 0)

    rpn_heads = RPNHeads(256 * 4, anchor_generator.num_anchors_per_location()[0])
    rpn_heads.load_state_dict(rpn_state_dict)
    rpn = detection_model.RPN(rpn_heads, anchor_generator, box_selector)

    rpn_provider = detection_model.RPNProvider(backbone, rpn, box_sampler=None)

    device = torch.device('cuda')
    rpn_provider.to(device)


    from PIL import Image

    x = Image.open('/datasets01/COCO/060817/val2014/COCO_val2014_000000000139.jpg').convert('RGB').resize((1200, 800))
    y = Image.open('/datasets01/COCO/060817/val2014/COCO_val2014_000000000285.jpg').convert('RGB')

    import numpy as np
    x = torch.from_numpy(np.array(x)).float() - torch.tensor([102.9801, 115.9465, 122.7717])
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)

    # x = torch.rand(2, 3, 800, 1300, device=device)
    # x = [torch.rand(3, 800, 1300, device=device), torch.rand(3, 700, 1400, device=device)]

    pooler = Pooler()
    classifier_layers = resnet50_conv5_head(num_classes=81, pretrained=pretrained_path)
    postprocessor = PostProcessor()
    classifier_head = detection_model.ClassificationHead(pooler, classifier_layers, postprocessor)

    model = detection_model.GeneralizedRCNN(rpn_provider, classifier_head)
    model.to(device)
    
    o = model.predict(x)

    from IPython import embed; embed()
