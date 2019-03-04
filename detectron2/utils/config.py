import os
from yacs.config import CfgNode as _CfgNode
from yacs.config import load_cfg

RESERVED_KEY = "_BASE_"


def merge_file_into_cfg(cfg, config_file):
    """
    Like `cfg.merge_from_file(config_file)`, but support a reserved key "_BASE_".
    If "_BASE_" exists, it will be a path (either absolute or relative to config_file) to a
    base config file to load before loading config_file. The base config file can recursively
    have its own base as well.
    Args:
        cfg: a yacs CfgNode instance. Will be modified.
        config_file (str)
    """

    with open(config_file, "r") as f:
        loaded_cfg = load_cfg(f)

    if RESERVED_KEY in loaded_cfg:
        base_cfg_file = getattr(loaded_cfg, RESERVED_KEY)
        if not base_cfg_file.startswith("/"):
            # the path to base cfg is relative to the config file itself
            base_cfg_file = os.path.join(os.path.dirname(config_file), base_cfg_file)
        merge_file_into_cfg(cfg, base_cfg_file)
        del loaded_cfg[RESERVED_KEY]
    cfg.merge_from_other_cfg(loaded_cfg)


class CfgNode(_CfgNode):
    """
    Similar to :class:`yacs.config.CfgNode`, but overwrite `merge_from_file` to support
    the "_BASE_" key.
    """

    def merge_from_file(self, cfg_filename):
        return merge_file_into_cfg(self, cfg_filename)

    def merge_from_other_cfg(self, cfg_other):
        assert (
            RESERVED_KEY not in cfg_other
        ), "The reserved key '{}' can only be used in files!".format(RESERVED_KEY)
        return super().merge_from_other_cfg(cfg_other)

    def merge_from_list(self, cfg_list):
        keys = set(cfg_list[0::2])
        assert RESERVED_KEY not in keys, "The reserved key '{}' can only be used in files!".format(
            RESERVED_KEY
        )
        return super().merge_from_list(cfg_list)
