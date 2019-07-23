# -*- coding: utf-8 -*-

from borc.common.config import CfgNode as _CfgNode


class CfgNode(_CfgNode):
    """
    The same as `borc.common.config.CfgNode`, but different in:

    1. Use unsafe yaml loading by default.
       Note that this may lead to arbitrary code execution: you must not
       load a config file from untrusted sources before manually inspecting
       the content of the file.
    2. TODO: convert old version of config files.
    """

    def merge_from_file(self, cfg_filename: str, allow_unsafe=True):
        super().merge_from_file(cfg_filename, allow_unsafe)


global_cfg = CfgNode()


def get_cfg():
    """
    Get a copy of the default config.

    Returns:
        a detectron2 CfgNode instance.
    """
    from .defaults import _C

    return _C.clone()


def set_global_cfg(cfg):
    """
    Let the global config point to the given cfg.

    Assume that the given "cfg" has the key "KEY", after calling
    `set_global_cfg(cfg)`, the key can be accessed by:

        from detectron2.config import global_cfg
        print(global_cfg.KEY)

    By using a hacky global config, you can access these configs anywhere,
    without having to pass the config object or the values deep into the code.
    This is a hacky feature introduced for quick prototyping / research exploration.
    """
    global global_cfg
    global_cfg.clear()
    global_cfg.update(cfg)
