import logging
from collections import OrderedDict

import torch


def align_and_update_state_dicts(model_state_dict, loaded_state_dict):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    model_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))

    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [len(j) if i.endswith(j) else 0 for i in model_keys for j in loaded_keys]
    match_matrix = torch.as_tensor(match_matrix).view(len(model_keys), len(loaded_keys))
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in model_keys]) if model_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    logger = logging.getLogger(__name__)
    matched_model_keys = set()
    matched_loaded_keys = set()
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = model_keys[idx_new]
        key_old = loaded_keys[idx_old]
        loaded_value = loaded_state_dict[key_old]
        shape_in_model = model_state_dict[key].shape

        if shape_in_model != loaded_value.shape:
            raise ValueError(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}".format(
                    key_old, loaded_value.shape, key, shape_in_model
                )
            )

        model_state_dict[key] = loaded_value
        matched_model_keys.add(key)
        matched_loaded_keys.add(key_old)
        logger.info(
            log_str_template.format(key, max_size, key_old, max_size_loaded, tuple(shape_in_model))
        )
    # print warnings about unmatched keys on both side
    unmatched_model_keys = [k for k in model_keys if k not in matched_model_keys]
    logger.warn(
        "Keys in the model but not loaded from checkpoint: " + ", ".join(unmatched_model_keys)
    )

    unmatched_loaded_keys = [k for k in loaded_keys if k not in matched_loaded_keys]
    logger.warn(
        "Keys in the checkpoint but not found in the model: " + ", ".join(unmatched_loaded_keys)
    )


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def load_state_dict(model, loaded_state_dict):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict)

    # use strict loading
    model.load_state_dict(model_state_dict)
