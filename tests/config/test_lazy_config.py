# Copyright (c) Facebook, Inc. and its affiliates.
import os
import unittest
import tempfile
from itertools import count

from detectron2.config import LazyConfig


class TestLazyPythonConfig(unittest.TestCase):
    def setUp(self):
        self.root_filename = os.path.join(os.path.dirname(__file__), "root_cfg.py")

    def test_load(self):
        cfg = LazyConfig.load(self.root_filename)

        self.assertEqual(cfg.dir1a_dict.a, "modified")
        self.assertEqual(cfg.dir1b_dict.a, 1)
        self.assertEqual(cfg.lazyobj.x, "base_a_1")

        cfg.lazyobj.x = "new_x"
        # reload
        cfg = LazyConfig.load(self.root_filename)
        self.assertEqual(cfg.lazyobj.x, "base_a_1")

    def test_save_load(self):
        cfg = LazyConfig.load(self.root_filename)
        with tempfile.TemporaryDirectory(prefix="detectron2") as d:
            fname = os.path.join(d, "test_config.yaml")
            LazyConfig.save(cfg, fname)
            cfg2 = LazyConfig.load(fname)

        self.assertEqual(cfg2.lazyobj._target_, "itertools.count")
        self.assertEqual(cfg.lazyobj._target_, count)
        cfg2.lazyobj.pop("_target_")
        cfg.lazyobj.pop("_target_")
        # the rest are equal
        self.assertEqual(cfg, cfg2)

    def test_overrides(self):
        cfg = LazyConfig.load(self.root_filename)
        LazyConfig.apply_overrides(cfg, ["lazyobj.x=123", 'dir1b_dict.a="123"'])
        self.assertEqual(cfg.dir1b_dict.a, "123")
        self.assertEqual(cfg.lazyobj.x, 123)
