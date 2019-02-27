import os
import PIL
from torch.utils.collect_env import get_pretty_env_info


def get_pil_version():
    return "\n        Pillow ({})".format(PIL.__version__)


def get_env_module():
    var_name = "DETECTRON2_ENV_MODULE"
    return "{}: {}".format(var_name, os.environ.get(var_name, "<not set>"))


def collect_env_info():
    env_str = get_env_module() + "\n"
    env_str += get_pretty_env_info()
    env_str += get_pil_version()
    return env_str
