from detectron2.utils.config import CfgNode

from .defaults import _C


global_cfg = CfgNode()


def get_cfg():
    """
    Get a copy of the default config.

    Returns:
        a detectron2 CfgNode instance.
    """
    return _C.clone()


def set_global_cfg(cfg):
    global global_cfg
    global_cfg.clear()
    global_cfg.update(cfg)
