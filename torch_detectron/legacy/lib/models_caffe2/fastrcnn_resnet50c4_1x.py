from collections import OrderedDict
import pickle

import torch
from torchvision import models
from torch.autograd import Variable
from torch import nn
import math

from ..layers import ROIAlign, ROIPool

"""
import torch.cuda
_ = torch.rand(2, 2).cuda()
# _ = _ @ _
from torch.utils.cpp_extension import load as load_ext
mod_path = '/private/home/fmassa/github/detectron.pytorch/torch_detectron/lib/layers/'
loaded_mod = load_ext('torch_cuda_roi_pool', [mod_path + 'ROIPool_cuda.cpp', mod_path + 'ROIPool.cu'], extra_cflags=['-O2'])

def roi_pooling(input, rois, img_idxs, size=(14,14), spatial_scale=1.0 / 16):
    boxes = torch.cat([Variable(img_idxs[:, None].float().cuda()), rois], 1)
    o = loaded_mod.roi_pool_cuda(input.data, boxes.data, spatial_scale, size[0], size[1])
    return Variable(o)

def roi_align(input, rois, img_idxs, size=(14,14), spatial_scale=1.0 / 16, sampling_ratio=0):
    boxes = torch.cat([Variable(img_idxs[:, None].float().cuda()), rois], 1)
    o = loaded_mod.roi_align_cuda(input.data, boxes.data, spatial_scale, size[0], size[1], sampling_ratio)
    return Variable(o)
"""

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


class ResNet50_FastRCNN(nn.Module):
    def __init__(self, weights=None, num_classes=81, **kwargs):
        self.inplanes = 64
        block = Bottleneck
        layers = [3, 4, 6, 3]
        super(ResNet50_FastRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv1_bn = FixedBatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res2 = self._make_layer(block, 64, layers[0])
        self.res3 = self._make_layer(block, 128, layers[1], stride=2)
        self.res4 = self._make_layer(block, 256, layers[2], stride=2)
        self.roi_pool = ROIAlign(output_size=(14, 14),
                spatial_scale=1.0 / 16, sampling_ratio=0)
        # self.roi_pool = ROIPool(output_size=(14, 14),
        #         spatial_scale=1.0 / 16)
        self._roi_pool = self.roi_pool
        self.res5 = self._make_layer(block, 512, layers[3], stride=2, return_out=False)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.cls_score = nn.Linear(512 * block.expansion, num_classes)
        self.bbox_pred = nn.Linear(512 * block.expansion, num_classes * 4)

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
        x = x.detach()

        x = self.res3(x)
        x = self.res4(x)
        return x

    def _classifier(self, x):
        res5 = self.res5(x)

        avgpool = self.avgpool(res5)
        x = avgpool.view(res5.size(0), -1)

        score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)

        return score, bbox_pred

    def forward(self, img, proposals, img_idxs):

        """
        min_size = min(img.shape[2:])
        max_size = max(img.shape[2:])
        MIN_SIZE = 600.  # 800.
        MAX_SIZE = 1000.  # 1333.
        ratio = MIN_SIZE / min_size
        if max_size * ratio > MAX_SIZE:
            ratio = MAX_SIZE / max_size
        sizes = [int(math.ceil(img.shape[2] * ratio)), int(math.ceil(img.shape[3] * ratio))]
        img = torch.nn.functional.upsample(img, size=sizes, mode='bilinear')
        proposals = proposals * ratio
        """

        x = self.conv1(img)
        x = self.conv1_bn(x)
        x = self.relu(x)
        xx = x.clone()
        x = self.maxpool(x)

        res2 = self.res2(x)
        res2 = res2.detach()

        res3 = self.res3(res2)
        res3.retain_grad()
        res4 = self.res4(res3)
        res4.retain_grad()
        # x = roi_align(x, proposals, img_idxs)

        # print('before roi', x.data.max(), x.data.norm())
        proposals = torch.cat([img_idxs.view(-1, 1).float(), proposals], 1)
        pool5 = self.roi_pool(res4, proposals)
        # x = roi_align(x, proposals, img_idxs.data)
        pool5.retain_grad()

        # print('after roi', x.data.max(), x.data.norm())

        # res5, res5_2 = self.res5(pool5)
        res5 = self.res5(pool5)
        res5.retain_grad()
        # res5_2.retain_grad()

        avgpool = self.avgpool(res5)
        avgpool.retain_grad()
        x = avgpool.view(res5.size(0), -1)
        
        # print('final', x.data.max(), x.data.norm())
        # x = x.detach()
        score = self.cls_score(x)
        # score.retain_grad()
        # score = torch.nn.functional.softmax(score, dim=-1)
        bbox_pred = self.bbox_pred(x)

        # return score.data, bbox_pred.data
        # return score, bbox_pred, avgpool, xx, res2, res3, pool5, res4
        return score, bbox_pred
        # return score, bbox_pred, pool5, res3, res4, res5, avgpool, res5_2


def load_resnet50_model(file_path='/private/home/fmassa/model_final.pkl'):
    return ResNet50_FastRCNN(file_path)

