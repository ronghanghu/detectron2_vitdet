from collections import OrderedDict
import pickle

import torch
from torchvision import models
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import math

from ..layers import ROIAlign, ROIPool


class FixedBatchNorm2d(nn.Module):
    def __init__(self, n):
        super(FixedBatchNorm2d, self).__init__()
        self.register_buffer('scale', torch.zeros(1, n, 1, 1))
        self.register_buffer('bias', torch.zeros(1, n, 1, 1))

    def forward(self, x):
        return x * Variable(self.scale) + Variable(self.bias)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, return_out=False):
        super(Bottleneck, self).__init__()
        # TODO had to modify the stride because of difference in implementation.
        # Invert compared to pytorch model
        self.branch2a = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.branch2a_bn = FixedBatchNorm2d(planes)
        self.branch2b = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.branch2b_bn = FixedBatchNorm2d(planes)
        self.branch2c = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.branch2c_bn = FixedBatchNorm2d(planes * 4)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        if downsample is not None:
            self.branch1 = nn.Conv2d(inplanes, planes * self.expansion,
                                    kernel_size=1, stride=stride, bias=False)
            self.branch1_bn = FixedBatchNorm2d(planes * self.expansion)
        self.stride = stride
        self.return_out = return_out

    def forward(self, x):
        residual = x

        out = self.branch2a(x)
        out = self.branch2a_bn(out)
        out = self.relu(out)

        out = self.branch2b(out)
        out = self.branch2b_bn(out)
        out = self.relu(out)

        out0 = self.branch2c(out)
        out = self.branch2c_bn(out0)

        if self.downsample is not None:
            residual = self.branch1(x)
            residual = self.branch1_bn(residual)

        out += residual
        out = self.relu(out)

        if self.return_out:
            return out, out0
        return out


class ResNet50_FPN(nn.Module):
    def __init__(self, weights=None, num_classes=81, **kwargs):
        self.inplanes = 64
        block = Bottleneck
        layers = [3, 4, 6, 3]
        super(ResNet50_FPN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv1_bn = FixedBatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res2 = self._make_layer(block, 64, layers[0])
        self.res3 = self._make_layer(block, 128, layers[1], stride=2)
        self.res4 = self._make_layer(block, 256, layers[2], stride=2)

        # TODO wrong, need to chenge the scale
        self.roi_pool = ROIAlign(output_size=(14, 14),
                spatial_scale=1.0 / 16, sampling_ratio=2)
        # self.roi_pool = ROIPool(output_size=(14, 14),
        #         spatial_scale=1.0 / 16)
        self._roi_pool = self.roi_pool
        self.res5 = self._make_layer(block, 512, layers[3], stride=2, return_out=False)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

        self.fpn_inner1 = nn.Conv2d(64 * 4, 256, 1)
        self.fpn_inner2 = nn.Conv2d(128 * 4, 256, 1)
        self.fpn_inner3 = nn.Conv2d(256 * 4, 256, 1)
        self.fpn_inner4 = nn.Conv2d(512 * 4, 256, 1)

        self.fpn_layer1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.fpn_layer2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.fpn_layer3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.fpn_layer4 = nn.Conv2d(256, 256, 3, 1, 1)

        self.fc6 = nn.Linear(256 * 7 * 7, 1024)
        self.fc7 = nn.Linear(1024, 1024)

        # Mask RCNN specific
        self.mask_fcn1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.mask_fcn2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.mask_fcn3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.mask_fcn4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        self.conv5_mask = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.mask_fcn_logits = nn.Conv2d(256, num_classes, 1)

        import torch.nn.init
        for l in [self.fpn_inner1, self.fpn_inner2, self.fpn_inner3, self.fpn_inner4,
                self.fpn_layer1, self.fpn_layer2, self.fpn_layer3, self.fpn_layer4]:
            torch.nn.init.kaiming_uniform_(l.weight, a=1)
            torch.nn.init.constant_(l.bias, 0)

        # TODO this was missing from the Faster RCNN, ADD IT!!!
        for l in [self.cls_score, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

        for l in [self.mask_fcn1, self.mask_fcn2, self.mask_fcn3, self.mask_fcn4, self.conv5_mask, self.mask_fcn_logits]:
            torch.nn.init.kaiming_normal_(l.weight, mode='fan_out')
            torch.nn.init.constant_(l.bias, 0)


        if weights is not None:
            if isinstance(weights, str):  # TODO py2 compat
                with open(weights, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                weights = data['blobs']
            weights = self._parse_weights(weights)
            self.load_state_dict(weights, strict=False)

        for p in self.conv1.parameters(): p.requires_grad = False
        for p in self.res2.parameters(): p.requires_grad = False
        if False:
            for m in [self.res3, self.res4]:
                for p in m.parameters():
                    p.requires_grad = False

    def _parse_weights(self, weights):
        original_keys = sorted(weights.keys())
        layer_keys = sorted(weights.keys())
        layer_keys = [k.replace('_', '.') for k in layer_keys]
        layer_keys = [k.replace('.w', '.weight') for k in layer_keys]
        layer_keys = [k.replace('.bn', '_bn') for k in layer_keys]
        layer_keys = [k.replace('.b', '.bias') for k in layer_keys]
        layer_keys = [k.replace('_bn.s', '_bn.scale') for k in layer_keys]
        layer_keys = [k.replace('.biasranch', '.branch') for k in layer_keys]
        layer_keys = [k.replace('bbox.pred', 'bbox_pred') for k in layer_keys]
        layer_keys = [k.replace('cls.score', 'cls_score') for k in layer_keys]
        layer_keys = [k.replace('res.conv1_', 'conv1_') for k in layer_keys]

        # RPN / Faster RCNN
        layer_keys = [k.replace('.biasbox', '.bbox') for k in layer_keys]
        layer_keys = [k.replace('conv.rpn', 'rpn.conv') for k in layer_keys]
        layer_keys = [k.replace('rpn.bbox.pred', 'rpn.bbox_pred') for k in layer_keys]
        layer_keys = [k.replace('rpn.cls.logits', 'rpn.cls_logits') for k in layer_keys]


        # FPN
        layer_keys = [k.replace('fpn.inner.res2.2.sum.lateral', 'fpn_inner1') for k in layer_keys]
        layer_keys = [k.replace('fpn.inner.res3.3.sum.lateral', 'fpn_inner2') for k in layer_keys]
        layer_keys = [k.replace('fpn.inner.res4.5.sum.lateral', 'fpn_inner3') for k in layer_keys]
        layer_keys = [k.replace('fpn.inner.res5.2.sum', 'fpn_inner4') for k in layer_keys]


        layer_keys = [k.replace('fpn.res2.2.sum', 'fpn_layer1') for k in layer_keys]
        layer_keys = [k.replace('fpn.res3.3.sum', 'fpn_layer2') for k in layer_keys]
        layer_keys = [k.replace('fpn.res4.5.sum', 'fpn_layer3') for k in layer_keys]
        layer_keys = [k.replace('fpn.res5.2.sum', 'fpn_layer4') for k in layer_keys]

        layer_keys = [k.replace('rpn.conv.fpn2', 'rpn.conv') for k in layer_keys]
        layer_keys = [k.replace('rpn.bbox_pred.fpn2', 'rpn.bbox_pred') for k in layer_keys]
        layer_keys = [k.replace('rpn.cls_logits.fpn2', 'rpn.cls_logits') for k in layer_keys]

        # Mask R-CNN
        layer_keys = [k.replace('mask.fcn.logits', 'mask_fcn_logits') for k in layer_keys]
        layer_keys = [k.replace('.[mask].fcn', 'mask_fcn') for k in layer_keys]
        layer_keys = [k.replace('conv5.mask', 'conv5_mask') for k in layer_keys]

        # from IPython import embed; embed()

        key_map = {k:v for k, v in zip(original_keys, layer_keys)}

        new_weights = OrderedDict()
        for k, v in weights.items():
            if '.momentum' in k:
                continue
            # if 'fc1000' in k:
            #     continue
            # new_weights[key_map[k]] = torch.from_numpy(v)
            w = torch.from_numpy(v)
            if 'bn' in k:
                w = w.view(1, -1, 1, 1)
            new_weights[key_map[k]] = w

        return new_weights

    def _make_layer(self, block, planes, blocks, stride=1, return_out=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = True

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            ro = False
            if return_out and i == blocks - 1:
                ro = True
            layers.append(block(self.inplanes, planes, return_out=ro))

        return nn.Sequential(*layers)

    def _backbone(self, img):
        x = self.conv1(img)
        x = self.conv1_bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.res2(x)
        l1 = x
        x = x.detach()

        x = self.res3(x)
        l2 = x
        x = self.res4(x)
        l3 = x
        x = self.res5(x)
        l4 = x

        inner4 = self.fpn_inner4(l4)

        inner4_top_down = F.upsample(inner4, scale_factor=2, mode='nearest')
        inner3_lateral = self.fpn_inner3(l3)
        inner3 = inner3_lateral + inner4_top_down

        inner3_top_down = F.upsample(inner3, scale_factor=2, mode='nearest')
        inner2_lateral = self.fpn_inner2(l2)
        inner2 = inner2_lateral + inner3_top_down

        inner2_top_down = F.upsample(inner2, scale_factor=2, mode='nearest')
        inner1_lateral = self.fpn_inner1(l1)
        inner1 = inner1_lateral + inner2_top_down

        fpn_l1 = self.fpn_layer1(inner1)
        fpn_l2 = self.fpn_layer2(inner2)
        fpn_l3 = self.fpn_layer3(inner3)
        fpn_l4 = self.fpn_layer4(inner4)

        fpn_l5 = F.max_pool2d(fpn_l4, 1, 2, 0)  # use subsample?

        return fpn_l1, fpn_l2, fpn_l3, fpn_l4, fpn_l5


    def _classifier(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


    def _masker(self, x):
        x = F.relu(self.mask_fcn1(x))
        x = F.relu(self.mask_fcn2(x))
        x = F.relu(self.mask_fcn3(x))
        x = F.relu(self.mask_fcn4(x))

        # upsample
        x = self.conv5_mask(x)
        
        x = self.mask_fcn_logits(x)
        return x
