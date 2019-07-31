#!/usr/bin/env python


import os
import tempfile
import unittest

from detectron2.config import downgrade_config, get_cfg, upgrade_config

_V0_CFG = """
MODEL:
  RPN_HEAD:
    NAME: "TEST"
VERSION: 0
"""


class TestConfigVersioning(unittest.TestCase):
    def test_upgrade_downgrade_consistency(self):
        cfg = get_cfg()
        cfg.USER_CUSTOM = 1

        down = downgrade_config(cfg, to_version=0)
        up = upgrade_config(down)
        self.assertTrue(up == cfg)

    def test_auto_upgrade(self):
        cfg = get_cfg()
        cfg.USER_CUSTOM = 1

        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        try:
            f.write(_V0_CFG)
            f.close()
            cfg.merge_from_file(f.name)
        finally:
            os.remove(f.name)

        self.assertEqual(cfg.MODEL.RPN.HEAD_NAME, "TEST")
        self.assertEqual(cfg.VERSION, 1)
