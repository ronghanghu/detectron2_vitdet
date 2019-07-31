# -*- coding: utf-8 -*-


import unittest

from detectron2.data import transforms as T
from detectron2.utils.logger import setup_logger


class TestTransforms(unittest.TestCase):
    def setUp(self):
        setup_logger()

    def test_register(self):
        dtype = "int"

        def add1(t, x):
            return x + 1

        def flip_sub_width(t, x):
            return x - t.width

        T.Transform.register_type(dtype, add1)
        T.HFlipTransform.register_type(dtype, flip_sub_width)

        transforms = T.TransformList(
            [
                T.ResizeTransform(0, 0, 0, 0, 0),  # +1
                T.CropTransform(0, 0, 0, 0),  # +1
                T.HFlipTransform(3),  # -3
            ]
        )
        self.assertEqual(transforms.apply_int(3), 2)

        with self.assertRaises(AssertionError):
            T.HFlipTransform.register_type(dtype, lambda x: 1)
