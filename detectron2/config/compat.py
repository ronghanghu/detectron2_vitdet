"""
Backward compatibility of configs.

Instructions to bump version:
+ It's not needed to bump version if new keys are added.
  It's only needed when backward-incompatible changes happen
  (i.e., some existing keys disappear, or the meaning of a key changes)
+ To bump version, do the following:
    1. Increment _C.VERSION in defaults.py
    2. Add a converter in this file.

      Each ConverterVX has a function "upgrade" which in-place upgrades config from X-1 to X,
      and a function "downgrade" which in-place downgrades config from X to X-1

      In each function, VERSION is left unchanged.

      Each converter assumes that its input has the relevant keys
      (i.e., the input is not a partial config).
    3. Run the tests (test_config.py) to make sure the upgrade & downgrade
       functions are consistent.
"""

from .config import CfgNode as CN
from .defaults import _C

__all__ = ["upgrade_config", "downgrade_config"]


def upgrade_config(cfg, to_version=None):
    """
    Upgrade a config from its current version to a newer version.

    Args:
        cfg (CfgNode),
        to_version (int): defaults to the latest version.
    """
    cfg = cfg.clone()
    if to_version is None:
        to_version = _C.VERSION

    assert cfg.VERSION <= to_version, "Cannot upgrade from v{} to v{}!".format(
        cfg.VERSION, to_version
    )
    for k in range(cfg.VERSION, to_version):
        converter = globals()["ConverterV" + str(k + 1)]
        converter.upgrade(cfg)
        cfg.VERSION = k + 1
    return cfg


def downgrade_config(cfg, to_version):
    """
    Downgrade a config from its current version to an older version.

    Args:
        cfg (CfgNode),
        to_version (int):

    Note:
        A general downgrade of arbitrary configs is not always possible due to the
        different functionailities in different versions.
        The purpose of downgrade is only to recover the defaults in old versions,
        allowing it to load an old partial yaml config.
        Therefore, the implementation only needs to fill in the default values
        in the old version when a general downgrade is not possible.
    """
    cfg = cfg.clone()
    assert cfg.VERSION >= to_version, "Cannot downgrade from v{} to v{}!".format(
        cfg.VERSION, to_version
    )
    for k in range(cfg.VERSION, to_version, -1):
        converter = globals()["ConverterV" + str(k)]
        converter.downgrade(cfg)
        cfg.VERSION = k - 1
    return cfg


class ConverterV1:
    @staticmethod
    def upgrade(cfg):
        cfg.MODEL.RPN.HEAD_NAME = cfg.MODEL.RPN_HEAD.NAME
        del cfg["MODEL"]["RPN_HEAD"]

    @staticmethod
    def downgrade(cfg):
        cfg.MODEL.RPN_HEAD = CN()
        cfg.MODEL.RPN_HEAD.NAME = cfg.MODEL.RPN.HEAD_NAME
        del cfg["MODEL"]["RPN"]["HEAD_NAME"]
