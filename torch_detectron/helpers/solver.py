"""
Utility functions for constructing the optimizers
"""
import torch

from torch_detectron.helpers.config_utils import ConfigClass


class _SGDOptimizer(ConfigClass):
    def __call__(self, model):
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = self.BASE_LR
            weight_decay = self.WEIGHT_DECAY
            if 'bias' in key:
                lr = self.BASE_LR_BIAS
                weight_decay = self.WEIGHT_DECAY_BIAS
            params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]

        optimizer = torch.optim.SGD(params, lr,
                                    momentum=self.MOMENTUM)
        return optimizer

class _WarmupMultiStepLR(ConfigClass):
    def __call__(self, optimizer):
        from torch_detectron.utils.lr_scheduler import WarmupMultiStepLR
        return WarmupMultiStepLR(optimizer, self.STEPS, self.GAMMA,
                warmup_factor=self.WARMUP_FACTOR, warmup_iters=self.WARMUP_ITERS,
                warmup_method=self.WARMUP_METHOD)
