import torch

from detectron2.solver.build import get_default_optimizer_params

from newconfig import LazyCall as L

SGD = L(torch.optim.SGD)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer. Currently the DefaultTrainer takes care of this automatically.
        weight_decay_norm=0.0
    ),
    lr=0.02,
    momentum=0.9,
    weight_decay=1e-4,
)
