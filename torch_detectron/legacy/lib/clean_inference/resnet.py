import torch
from torch import nn

from layers import ROIAlign, FixedBatchNorm2d


# NOTE had to modify the stride because of difference in implementation.
# Invert compared to pytorch model
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.branch2a = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.branch2a_bn = FixedBatchNorm2d(planes)
        self.branch2b = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.branch2b_bn = FixedBatchNorm2d(planes)
        self.branch2c = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.branch2c_bn = FixedBatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if downsample is not None:
            self.branch1 = nn.Conv2d(inplanes, planes * self.expansion,
                                    kernel_size=1, stride=stride, bias=False)
            self.branch1_bn = FixedBatchNorm2d(planes * self.expansion)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.branch2a(x)
        out = self.branch2a_bn(out)
        out = self.relu(out)

        out = self.branch2b(out)
        out = self.branch2b_bn(out)
        out = self.relu(out)

        out = self.branch2c(out)
        out = self.branch2c_bn(out)

        if self.downsample is not None:
            residual = self.branch1(x)
            residual = self.branch1_bn(residual)

        out += residual
        out = self.relu(out)

        return out


class ResNet50COCO(nn.Module):
    def __init__(self, **kwargs):
        self.inplanes = 64
        block = Bottleneck
        layers = [3, 4, 6, 3]
        num_classes = 81
        super(ResNet50COCO, self).__init__(**kwargs)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv1_bn = FixedBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res2 = self._make_layer(block, 64, layers[0])
        self.res3 = self._make_layer(block, 128, layers[1], stride=2)
        self.res4 = self._make_layer(block, 256, layers[2], stride=2)
        self._roi_pool = ROIAlign(output_size=(14, 14),
                spatial_scale=1.0 / 16, sampling_ratio=0)
        self.res5 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.cls_score = nn.Linear(512 * block.expansion, num_classes)
        self.bbox_pred = nn.Linear(512 * block.expansion, num_classes * 4)

        for p in self.conv1.parameters(): p.requires_grad = False
        for p in self.res2.parameters(): p.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = True

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _preprocess_image(self, image):
        mean = [102.9801, 115.9465, 122.7717]
        dtype = image.dtype
        device = image.device
        mean = torch.tensor(mean, dtype=dtype, device=device).view(3, 1, 1)
        return (image[[2, 1, 0]] * 255) - mean

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

