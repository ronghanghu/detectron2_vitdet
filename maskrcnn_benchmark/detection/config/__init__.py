from .defaults import _C


def get_cfg():
    """
    Get a copy of the default config.

    Returns:
        a yacs CfgNode instance.
    """
    return _C.clone()
