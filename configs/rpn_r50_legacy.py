"""
This is an example config file.

The goal is to have as much flexibility as possible.

WARNING: this is an initial POC, and needs iteration!
It currently is too much verbose
"""

import os
import random
from bisect import bisect_right

import torch
import torchvision

from torch_detectron.core.image_list import to_image_list

# dataset
ann_file = '/private/home/fmassa/coco_trainval2017/annotations/instances_val2017_mod.json'
data_dir = '/datasets01/COCO/060817/val2014/'

# model
pretrained_path = '/private/home/fmassa/github/detectron.pytorch/torch_detectron/core/models/r50.pth'
num_gpus = 8

# training
images_per_batch = 2
lr = 0.00125 * num_gpus * images_per_batch
weight_decay = 0.0001
momentum = 0.9
lr_steps = [60000, 80000]
lr_gamma = 0.1
num_workers = 4

# FIXME those are the only properties that needs to be visible
# change this
device = torch.device('cuda')
#number_of_epochs = 2
max_iter = 90000

save_dir = os.environ['SAVE_DIR'] if 'SAVE_DIR' in os.environ else ''
checkpoint = os.environ['CHECKPOINT_FILE'] if 'CHECKPOINT_FILE' in os.environ else ''
do_test = True

# function getters
# should implement get_data_loader, get_model, get_optimizer and get_scheduler

class ImageTransform(object):
    """
    Data transformations to be performed in the image and the targets.
    The normalization is specific for the C2 pretrained models
    """
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, x, target):
        F = torchvision.transforms.functional
        x = F.resize(x, 800, max_size=1333)
        target = target.resize(x.size)
        if random.random() < self.flip_prob:
            x = F.hflip(x)
            target = target.transpose(0)
        x = F.to_tensor(x)

        x = x[[2, 1, 0]] * 255
        x -= torch.tensor([102.9801, 115.9465, 122.7717]).view(3,1,1)

        return x, target

class Collator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """
    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        return images, targets

def get_dataset():
    from torch_detectron.datasets.coco import COCODataset
    # dataset = COCODataset(ann_file, data_dir, ImageTransform())

    valminusminival_ann_file = '/private/home/fmassa/coco_trainval2017/annotations/instances_valminusminival2017.json'
    from torch.utils.data.dataset import ConcatDataset

    ann_file = '/datasets01/COCO/060817/annotations/instances_train2014.json'
    data_dir = '/datasets01/COCO/060817/train2014/'
    val_data_dir = '/datasets01/COCO/060817/val2014/'
    dataset_train = COCODataset(ann_file, data_dir, ImageTransform())
    dataset_valminusminival = COCODataset(valminusminival_ann_file, val_data_dir, ImageTransform())
    dataset = ConcatDataset([dataset_train, dataset_valminusminival])

    ann_file = '/private/home/fmassa/coco_trainval2017/annotations/instances_val2017_mod.json'
    data_dir = '/datasets01/COCO/060817/val2014/'
    dataset_val = COCODataset(ann_file, data_dir, ImageTransform(flip_prob=0))
    return dataset, dataset_val

def get_data_loader(distributed=False):
    dataset, dataset_val = get_dataset()
    size_divisible = 0
    collator = Collator(size_divisible)

    import torch.utils.data
    from torch.utils.data.sampler import RandomSampler
    sampler = RandomSampler(dataset)
    if distributed:
        import torch.utils.data.distributed
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=images_per_batch,
            num_workers=num_workers, sampler=sampler, collate_fn=collator)

    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1,
            num_workers=num_workers, shuffle=False, collate_fn=collator)

    return data_loader, data_loader_val


class RPNModel(torch.nn.Module):
    def __init__(self, pretrained_path):
        super(RPNModel, self).__init__()
        from torch_detectron.model_builder.resnet import resnet50_conv4_body
        from torch_detectron.core.anchor_generator import AnchorGenerator
        from torch_detectron.core.box_selector import RPNBoxSelector
        from torch_detectron.core.rpn_losses import (RPNLossComputation,
                RPNTargetPreparator)
        from torch_detectron.core.proposal_matcher import Matcher
        from torch_detectron.core.balanced_positive_negative_sampler import (
                BalancedPositiveNegativeSampler)
        from torch_detectron.core.faster_rcnn import RPNHeads
        from torch_detectron.core.box_coder import BoxCoder

        backbone = resnet50_conv4_body(pretrained_path)

        anchor_generator = AnchorGenerator(scales=(0.125, 0.25, 0.5, 1., 2.))
        rpn_heads = RPNHeads(256 * 4, anchor_generator.num_anchors_per_location()[0])

        matcher = Matcher(0.7, 0.3, force_match_for_each_row=True)
        box_coder = BoxCoder(weights=(1., 1., 1., 1.))
        target_preparator = RPNTargetPreparator(matcher, box_coder)
        fg_bg_sampler = BalancedPositiveNegativeSampler(
                batch_size_per_image=256, positive_fraction=0.5)
        self.rpn_loss_evaluator = RPNLossComputation(target_preparator, fg_bg_sampler)

        # for inference
        self.box_selector = RPNBoxSelector(12000, 2000, 0.7, 0)

        self.backbone = backbone
        self.anchor_generator = anchor_generator
        self.rpn_heads = rpn_heads

    def forward(self, images, targets=None):
        images = to_image_list(images)
        features = self.backbone(images.tensors)

        objectness, rpn_box_regression = self.rpn_heads(features)
        anchors = self.anchor_generator(images.image_sizes, features)

        if not self.training:
            result = self.box_selector(anchors, objectness, rpn_box_regression)
            return result[0]  # returns result for single feature map

        loss_objectness, loss_box_reg = self.rpn_loss_evaluator(
                anchors, objectness, rpn_box_regression, targets)

        return dict(
                loss_objectness=loss_objectness,
                loss_box_reg=loss_box_reg)

    def predict(self, images):
        return self(images)

def get_model():
    model = RPNModel(pretrained_path)
    model.to(device)
    return model

def get_optimizer(model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if 'bias' in key:
            params += [{'params': [value], 'lr': lr * (1 + 1), 'weight_decay': 0}]
        else:
            params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]

    optimizer = torch.optim.SGD(params, lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    return optimizer

# def get_scheduler(optimizer):
#     return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_gamma)

def get_lr_at_iter(it):
    """Get the learning rate at iteration it according to the cfg.SOLVER
    settings.
    """
    WARM_UP_FACTOR = 1.0 / 3
    WARM_UP_ITERS = 500
    WARM_UP_METHOD = 'linear'
    GAMMA = lr_gamma
    STEPS = lr_steps
    # lr = get_lr_func()(it)
    lr = GAMMA ** bisect_right(STEPS, it)
    if it < WARM_UP_ITERS:
        method = WARM_UP_METHOD
        if method == 'constant':
            warmup_factor = WARM_UP_FACTOR
        elif method == 'linear':
            alpha = it / WARM_UP_ITERS
            warmup_factor = WARM_UP_FACTOR * (1 - alpha) + alpha
        else:
            raise KeyError('Unknown SOLVER.WARM_UP_METHOD: {}'.format(method))
        lr *= warmup_factor
    return lr

def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_at_iter)
