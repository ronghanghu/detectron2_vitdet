import os
from yacs.config import CfgNode as _CfgNode

RESERVED_KEY = "_BASE_"


class CfgNode(_CfgNode):
    """
    Our own extended version of :class:`yacs.config.CfgNode`.
    It contains the following extra features:

    1. The :meth:`merge_from_file` method supports the "_BASE_" key.
    2. Keys that start with "COMPUTED_" are treated as insertion-only "computed" attributes.
       They can be inserted regardless of whether the CfgNode is frozen or not.
    """

    def merge_from_file(self, cfg_filename):
        with open(cfg_filename, "r") as f:
            loaded_cfg = self.load_cfg(f)

        if RESERVED_KEY in loaded_cfg:
            base_cfg_file = getattr(loaded_cfg, RESERVED_KEY)
            if not base_cfg_file.startswith("/"):
                # the path to base cfg is relative to the config file itself
                base_cfg_file = os.path.join(os.path.dirname(cfg_filename), base_cfg_file)
            self.merge_from_file(base_cfg_file)
            del loaded_cfg[RESERVED_KEY]
        self.merge_from_other_cfg(loaded_cfg)

    # Forward the following calls to base, but with a check on the RESERVED_KEY
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

    def __setattr__(self, name, val):
        if name.startswith("COMPUTED_"):
            if name in self:
                raise KeyError("Computed attributed {} already exists!".format(name))
            self[name] = val
        else:
            super().__setattr__(name, val)
