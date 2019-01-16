from maskrcnn_benchmark.utils.config import CfgNode

from .defaults import _C


def get_cfg():
    """
    Get a copy of the default config.

    Returns:
        a yacs CfgNode instance.
    """
    return CfgNode(_C.clone())
