import copy
import pickle
import re


def load_c2_weights(file_path):
    """
    Returns:
        dict: name -> numpy array
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    if "blobs" in data:
        weights = data["blobs"]
    else:
        weights = data
    weights = {k: v for k, v in weights.items() if not k.endswith("_momentum")}
    return weights


def convert_basic_c2_names(original_keys):
    """
    Apply some basic name conversion to names in C2 weights.
    It only deals with typical backbone models.

    Args:
        original_keys (list[str]):
    Returns:
        list[str]: The same number of strings matching those in original_keys.
    """
    layer_keys = copy.deepcopy(original_keys)
    layer_keys = [
        {"pred_b": "fc1000_b", "pred_w": "fc1000_w"}.get(k, k) for k in layer_keys
    ]  # some hard-coded mappings

    layer_keys = [k.replace("_", ".") for k in layer_keys]
    layer_keys = [re.sub(".b$", ".bias", k) for k in layer_keys]
    layer_keys = [re.sub(".w$", ".weight", k) for k in layer_keys]
    layer_keys = [re.sub("bn.s$", "bn.weight", k) for k in layer_keys]

    # stages
    layer_keys = [re.sub("^res.conv1.bn.", "bn1.", k) for k in layer_keys]
    layer_keys = [re.sub("^res2.", "layer1.", k) for k in layer_keys]
    layer_keys = [re.sub("^res3.", "layer2.", k) for k in layer_keys]
    layer_keys = [re.sub("^res4.", "layer3.", k) for k in layer_keys]
    layer_keys = [re.sub("^res5.", "layer4.", k) for k in layer_keys]

    # blocks
    layer_keys = [k.replace(".branch1.bn.", ".downsample.1.") for k in layer_keys]
    layer_keys = [k.replace(".branch1.", ".downsample.0.") for k in layer_keys]
    layer_keys = [k.replace(".branch2a.bn.", ".bn1.") for k in layer_keys]
    layer_keys = [k.replace(".branch2a.", ".conv1.") for k in layer_keys]
    layer_keys = [k.replace(".branch2b.bn.", ".bn2.") for k in layer_keys]
    layer_keys = [k.replace(".branch2b.", ".conv2.") for k in layer_keys]
    layer_keys = [k.replace(".branch2c.bn.", ".bn3.") for k in layer_keys]
    layer_keys = [k.replace(".branch2c.", ".conv3.") for k in layer_keys]
    return layer_keys
