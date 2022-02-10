import torch
import copy

from typing import Any, Dict, List, Optional, Set


def get_layer_id(layer_name, num_layers):
    if layer_name.startswith("backbone"):
        if "net.pos_embed" in layer_name:
            return 0
        elif "net.patch_embed" in layer_name:
            return 0
        elif "net.blocks." in layer_name:
            layer_id = int(layer_name[layer_name.find("net.blocks."):].split(".")[2])
            return layer_id + 1
    
    return num_layers - 1


def get_layer_decay_optimizer_params(
    model: torch.nn.Module,
    base_lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    weight_decay_norm: Optional[float] = None,
    bias_lr_factor: Optional[float] = 1.0,
    weight_decay_bias: Optional[float] = None,
    weight_decay_1d: Optional[float] = None,
    lr_decay_rate: Optional[float] = None,
    num_layers: Optional[int] = None,
    skip_lr_decay: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
):
    """
    Get default param list for optimizer, with support for a few types of
    overrides. If no overrides needed, this is equivalent to `model.parameters()`.

    Args:
        base_lr: lr for every group by default. Can be omitted to use the one in optimizer.
        weight_decay: weight decay for every group by default. Can be omitted to use the one
            in optimizer.
        weight_decay_norm: override weight decay for params in normalization layers
        bias_lr_factor: multiplier of lr for bias parameters.
        weight_decay_bias: override weight decay for bias parameters
        overrides: if not `None`, provides values for optimizer hyperparameters
            (LR, weight decay) for module parameters with a given name; e.g.
            ``{"embedding": {"lr": 0.01, "weight_decay": 0.1}}`` will set the LR and
            weight decay values for all module parameters named `embedding`.

    For common detection models, ``weight_decay_norm`` is the only option
    needed to be set. ``bias_lr_factor,weight_decay_bias`` are legacy settings
    from Detectron1 that are not found useful.

    Example:
    ::
        torch.optim.SGD(get_default_optimizer_params(model, weight_decay_norm=0),
                       lr=0.01, weight_decay=1e-4, momentum=0.9)
    """
    if overrides is None:
        overrides = {}
    defaults = {}
    if base_lr is not None:
        defaults["lr"] = base_lr
    if weight_decay is not None:
        defaults["weight_decay"] = weight_decay
    bias_overrides = {}
    if bias_lr_factor is not None and bias_lr_factor != 1.0:
        # NOTE: unlike Detectron v1, we now by default make bias hyperparameters
        # exactly the same as regular weights.
        if base_lr is None:
            raise ValueError("bias_lr_factor requires base_lr")
        bias_overrides["lr"] = base_lr * bias_lr_factor
    if weight_decay_bias is not None:
        bias_overrides["weight_decay"] = weight_decay_bias
    if len(bias_overrides):
        if "bias" in overrides:
            raise ValueError("Conflicting overrides for 'bias'")
        overrides["bias"] = bias_overrides
    if lr_decay_rate is not None:
        assert num_layers is not None and base_lr is not None
        num_layers += 2

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            # print(module_name, module_param_name, value.shape)
            hyperparams = copy.copy(defaults)
            if isinstance(module, norm_module_types) and weight_decay_norm is not None:
                hyperparams["weight_decay"] = weight_decay_norm
            if len(value.shape) == 1 and weight_decay_1d is not None:
                hyperparams["weight_decay"] = weight_decay_1d
                # print(f"weight decay {module_name}, {module_param_name} = {weight_decay_1d}")
            if lr_decay_rate is not None:
                if not (skip_lr_decay is not None and any([skip in module_name for skip in skip_lr_decay])): 
                #     print(f"skip lr decay {module_name}")
                # else:
                    layer_id = get_layer_id(f"{module_name}.{module_param_name}", num_layers)
                    hyperparams["lr"] *= lr_decay_rate ** (num_layers - 1 - layer_id)
                    # print(module_name, module_param_name, layer_id, hyperparams["lr"])
                    
            hyperparams.update(overrides.get(module_param_name, {}))
            params.append({"params": [value], **hyperparams})
    return params