import torch
import itertools

from detectron2.config import LazyCall as L
from detectron2.solver.build import get_default_optimizer_params
from .layer_decay_optim import get_layer_decay_optimizer_params


class AdamWGradClip(torch.optim.AdamW):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        clip_norm_val=1.0,
        clip_norm_type=2.0,
    ):
        self.clip_norm_val = clip_norm_val
        self.clip_norm_type = clip_norm_type

        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)

    @torch.no_grad()
    def step(self, closure=None):
        all_params = itertools.chain(*[x["params"] for x in self.param_groups])
        torch.nn.utils.clip_grad_norm_(all_params, self.clip_norm_val, self.clip_norm_type)
        super().step(closure=closure)


AdamW = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        weight_decay_norm=0.0,
        overrides={
            "pos_embed": {"weight_decay": 0.0},
        },
    ),
    lr=0.00002,
    betas=(0.9, 0.999),
    weight_decay=0.1,
)


AdamW_GC = L(AdamWGradClip)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        weight_decay_norm=0.0,
        overrides={
            "pos_embed": {"weight_decay": 0.0},
        },
    ),
    lr=0.00002,
    betas=(0.9, 0.999),
    weight_decay=0.1,
    clip_norm_val=1.0,
    clip_norm_type=2.0,
)


AdamLayerDecay = L(torch.optim.AdamW)(
    params=L(get_layer_decay_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        base_lr="${..lr}",
        lr_decay_rate=0.9,
        weight_decay_norm=0.0,
        overrides={
            "pos_embed": {"weight_decay": 0.0},
        },
    ),
    lr=0.00002,
    betas=(0.9, 0.999),
    weight_decay=0.1,
)