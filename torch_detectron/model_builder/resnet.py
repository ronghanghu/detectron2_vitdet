import torch
from torch import nn
from torch.nn import functional as F


class FixedBatchNorm2d(nn.Module):
    def __init__(self, n):
        super(FixedBatchNorm2d, self).__init__()
        self.register_buffer("scale", torch.zeros(1, n, 1, 1))
        self.register_buffer("bias", torch.zeros(1, n, 1, 1))

    def forward(self, x):
        return x * self.scale + self.bias


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # TODO had to modify the stride because of difference in implementation.
        # Invert compared to pytorch model
        self.branch2a = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=stride, bias=False
        )
        self.branch2a_bn = FixedBatchNorm2d(planes)
        self.branch2b = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.branch2b_bn = FixedBatchNorm2d(planes)
        self.branch2c = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.branch2c_bn = FixedBatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if downsample is not None:
            self.branch1 = nn.Conv2d(
                inplanes,
                planes * self.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
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

        out0 = self.branch2c(out)
        out = self.branch2c_bn(out0)

        if self.downsample is not None:
            residual = self.branch1(x)
            residual = self.branch1_bn(residual)

        out += residual
        out = self.relu(out)

        return out


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


class ResNetBackbone(nn.Module):
    def __init__(self, block, layers):
        """
        layers encode:
            - index of the block
            - number of blocks
            - if the result of the block should be returned or not
        """
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_bn = FixedBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.blocks = []
        self.should_return = {}
        for block_id, layer_config, should_return in layers:
            name = "res" + str(block_id)
            module = _make_layer(
                self,
                block,
                16 * 2 ** block_id,
                layer_config,
                stride=int(block_id > 2) + 1,
            )
            self.add_module(name, module)
            self.blocks.append(name)
            self.should_return[name] = should_return

        # TODO parametrize
        for p in self.conv1.parameters():
            p.requires_grad = False
        for p in self.res2.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outputs = []
        for block in self.blocks:
            x = getattr(self, block)(x)
            if self.should_return[block]:
                outputs.append(x)

        return outputs


class ResNetHead(nn.Module):
    def __init__(self, block, layers, stride_init=None):
        self.inplanes = 1024  # TODO make it generic
        super(ResNetHead, self).__init__()

        self.blocks = []
        stride = stride_init
        for block_id, layer_config in layers:
            name = "res" + str(block_id)
            if not stride:
                stride = int(block_id > 2) + 1
            module = _make_layer(
                self, block, 16 * 2 ** block_id, layer_config, stride=stride
            )
            stride = None
            self.add_module(name, module)
            self.blocks.append(name)

    def forward(self, x):
        for block in self.blocks:
            x = getattr(self, block)(x)
        return x


# TODO find a better name for this
class ClassifierHead(nn.Module):
    def __init__(self, num_inputs, num_classes):
        super(ClassifierHead, self).__init__()
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.bbox_pred = nn.Linear(num_inputs, num_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.weight, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.weight, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred


def resnet50_conv4_body(pretrained=None):
    model = ResNetBackbone(
        Bottleneck, layers=[(2, 3, False), (3, 4, False), (4, 6, True)]
    )
    if pretrained:
        state_dict = torch.load(pretrained)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet50_conv5_head(num_classes, pretrained=None):
    block = Bottleneck
    head = ResNetHead(block, layers=[(5, 3)])
    classifier = ClassifierHead(512 * block.expansion, num_classes)
    if pretrained:
        state_dict = torch.load(pretrained)
        head.load_state_dict(state_dict, strict=False)
        classifier.load_state_dict(state_dict, strict=False)
    model = nn.Sequential(head, classifier)
    return model


def resnet50_conv5_body():
    model = ResNetBackbone(Bottleneck, layers=[(2, 3), (3, 4), (4, 6), (5, 3)])
    return model


def fpn_resnet50_conv5_body(pretrained=None, **kwargs):
    from ..core.fpn import FPN, LastLevelMaxPool

    body = ResNetBackbone(
        Bottleneck, layers=[(2, 3, True), (3, 4, True), (4, 6, True), (5, 3, True)]
    )
    fpn = FPN(
        layers=[256, 512, 1024, 2048],
        representation_size=kwargs['REPRESENTATION_SIZE'],
        top_blocks=LastLevelMaxPool(),
    )
    if pretrained:
        state_dict = torch.load(pretrained)
        body.load_state_dict(state_dict, strict=False)
        fpn.load_state_dict(state_dict, strict=False)
    model = nn.Sequential(body, fpn)
    return model


def fpn_classification_head(num_classes, pretrained=None, **kwargs):
    from ..core.fpn import FPNHeadClassifier
    representation_size = kwargs['REPRESENTATION_SIZE']
    model = FPNHeadClassifier(num_classes, representation_size * 7 * 7, 1024)
    if pretrained:
        state_dict = torch.load(pretrained)
        model.load_state_dict(state_dict, strict=False)
    return model
