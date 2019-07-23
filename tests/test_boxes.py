import numpy as np
import unittest
import torch

from detectron2.structures import BoxMode


class TestBoxMode(unittest.TestCase):
    def _convert_xy_to_wh(self, x):
        return BoxMode.convert(x, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)

    def test_box_convert_list(self):
        for tp in [list, tuple]:
            box = tp([5, 5, 10, 10])
            output = self._convert_xy_to_wh(box)
            self.assertTrue(output == tp([5, 5, 5, 5]))

            with self.assertRaises(Exception):
                self._convert_xy_to_wh([box])

    def test_box_convert_array(self):
        box = np.asarray([[5, 5, 10, 10], [1, 1, 2, 3]])
        output = self._convert_xy_to_wh(box)
        self.assertTrue((output[0] == [5, 5, 5, 5]).all())
        self.assertTrue((output[1] == [1, 1, 1, 2]).all())

    def test_box_convert_tensor(self):
        box = torch.tensor([[5, 5, 10, 10], [1, 1, 2, 3]])
        output = self._convert_xy_to_wh(box).numpy()
        self.assertTrue((output[0] == [5, 5, 5, 5]).all())
        self.assertTrue((output[1] == [1, 1, 1, 2]).all())
