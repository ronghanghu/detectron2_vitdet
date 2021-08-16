import torch
import itertools

from detectron2.config import LazyCall as L
from detectron2.solver.build import get_default_optimizer_params

# SGD = L(torch.optim.SGD)(
#     params=L(get_default_optimizer_params)(
#         # params.model is meant to be set to the model object, before instantiating
#         # the optimizer.
#         weight_decay_norm=0.0
#     ),
#     lr=0.02,
#     momentum=0.9,
#     weight_decay=1e-4,
# )

def maybe_add_full_model_gradient_clipping(optim, params, lr, betas, weight_decay, clip_norm_val, clip_norm_type):
    # optim: the optimizer class
    # detectron2 doesn't have full model gradient clipping now
    print(clip_norm_val, clip_norm_type)
    class FullModelGradientClippingOptimizer(optim):
        def step(self, closure=None):
            all_params = itertools.chain(*[x["params"] for x in self.param_groups])
            torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val, clip_norm_type)
            super().step(closure=closure)

    return FullModelGradientClippingOptimizer(params, lr, betas, weight_decay)


class AdamWGC(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False,
                 clip_norm_val=1.0, clip_norm_type=2.0):
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
        }
    ),
    lr=0.00002,
    betas=(0.9, 0.999),
    weight_decay=0.1,
)

AdamWGradClip = L(maybe_add_full_model_gradient_clipping)(
    optim=torch.optim.AdamW,
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        weight_decay_norm=0.0,
        overrides={
            "pos_embed": {"weight_decay": 0.0},
        }
    ),
    lr=0.00002,
    betas=(0.9, 0.999),
    weight_decay=0.1,
    clip_norm_val=1.0,
    clip_norm_type=2.0,
)


AdamW_GC = L(AdamWGC)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        weight_decay_norm=0.0,
        overrides={
            "pos_embed": {"weight_decay": 0.0},
        }
    ),
    lr=0.00002,
    betas=(0.9, 0.999),
    weight_decay=0.1,
    clip_norm_val=1.0,
    clip_norm_type=2.0,
)
