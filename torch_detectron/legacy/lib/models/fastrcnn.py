from collections import OrderedDict
import math

import torch
from torch import nn
from torch.autograd import Variable

import torch.utils.model_zoo as model_zoo

from torchvision import models

from ..layers import ROIAlign, ROIPool, FixedBatchNorm2d


class FastRCNN(models.ResNet):
    def __init__(self, block, layers, num_classes):
        super(FastRCNN, self).__init__(block, layers, 1000)

        self.roi_pool = ROIAlign(output_size=(14, 14),
                spatial_scale=1.0 / 16, sampling_ratio=0)
        # self.roi_pool = ROIPool(output_size=(14, 14),
        #         spatial_scale=1.0 / 16)
        self.cls_score = nn.Linear(512 * block.expansion, num_classes)
        self.bbox_pred = nn.Linear(512 * block.expansion, num_classes * 4)

        # nn.init.normal(self.cls_score.weight, std=0.01)  # 0.01
        # nn.init.constant(self.cls_score.bias, 0)
        nn.init.normal(self.bbox_pred.weight, std=0.001)
        nn.init.constant(self.bbox_pred.bias, 0)


    def forward(self, img, proposals, img_idx):
        # from IPython import embed; embed()
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

        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = x.detach()

        x = self.layer2(x)
        
        # x = x.detach()

        x = self.layer3(x)

        # x = x.detach()

        # print(x.shape)
        # print('before roi', x.data.max(), x.data.norm())
        # proposals = torch.cat([img_idx.view(-1, 1).float(), proposals], 1)
        proposals = Variable(torch.cat([img_idx.data.view(-1, 1).float(), proposals.data], 1))
        x = self.roi_pool(x, proposals)

        # print('after roi', x.data.max(), x.data.norm(), x.shape)
        # x = x.detach()

        x = self.layer4(x)
        # print('after layer4', x.data.max(), x.data.norm(), x.shape)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # x = torch.nn.functional.dropout(x, 0.7, True)
        # x = x.detach()
        # print('final', x.data.max(), x.data.norm())

        # print('before class', x.data.min(), x.data.max())
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


class FastRCNNVGG(models.VGG):
    def __init__(self, features, num_classes):
        super(FastRCNNVGG, self).__init__(features, 1000)
        clss = list(self.classifier.children())
        # self.classifier = torch.nn.Sequential(clss[0], clss[1], clss[3], clss[4])
        self.classifier = torch.nn.Sequential(*clss[:-1])

        # self.roi_pool = ROIAlign(output_size=(14, 14),
        #         spatial_scale=1.0 / 16, sampling_ratio=0)
        self.roi_pool = ROIPool(output_size=(7, 7),
                spatial_scale=1.0 / 16)
        self.cls_score = nn.Linear(4096, num_classes)
        self.bbox_pred = nn.Linear(4096, num_classes * 4)

        nn.init.normal(self.cls_score.weight, std=0.01)  # 0.01
        nn.init.constant(self.cls_score.bias, 0)
        nn.init.normal(self.bbox_pred.weight, std=0.001)
        nn.init.constant(self.bbox_pred.bias, 0)


    def forward(self, img, proposals, img_idx):
        # from IPython import embed; embed()
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

        x = self.features(img)

        proposals = Variable(torch.cat([img_idx.data.view(-1, 1).float(), proposals.data], 1))
        x = self.roi_pool(x, proposals)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def _fix_batchnorm2d(model):
    # for name, m in model.named_children():
    for name, m in model.named_modules():  # TODO suboptimal maybe, but well
        for sub_m in m.children():
            _fix_batchnorm2d(sub_m)
        if isinstance(m, nn.BatchNorm2d):
            weight = m.weight.data
            bias = m.bias.data
            running_mean = m.running_mean
            running_var = m.running_var
            eps = m.eps
            inv_std = 1. / (running_var + eps).sqrt()

            scale = weight * inv_std
            bias = bias - running_mean * inv_std * weight
            fixed = FixedBatchNorm2d(scale.size(0))
            fixed.scale.copy_(scale.view(1, -1, 1, 1))
            fixed.bias.copy_(bias.view(1, -1, 1, 1))

            setattr(model, name, fixed)

def fastrcnn_resnet18(**kwargs):
    model = FastRCNN(models.resnet.BasicBlock, [2, 2, 2, 2], **kwargs)
    model_urls = models.resnet.model_urls
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    _fix_batchnorm2d(model)
    # _fix_batchnorm2d(model.layer2[0].downsample)
    # _fix_batchnorm2d(model.layer3[0].downsample)
    # _fix_batchnorm2d(model.layer4[0].downsample)
    print(model)
    return model

def fastrcnn_resnet50(**kwargs):
    model = FastRCNN(models.resnet.Bottleneck, [3, 4, 6, 3], **kwargs)
    model_urls = models.resnet.model_urls
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    _fix_batchnorm2d(model)
    return model

def fastrcnn_vgg16(**kwargs):
    cfg = models.vgg.cfg['D']
    cfg = cfg[:-1]  # remove last max pool
    model = FastRCNNVGG(models.vgg.make_layers(cfg), **kwargs)
    model_urls = models.vgg.model_urls
    model.load_state_dict(model_zoo.load_url(model_urls['vgg16']), strict=False)
    return model
