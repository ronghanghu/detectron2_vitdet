import torch_detectron as td
import unittest

class Tester(unittest.TestCase):

    def test_bbox_hflip(self):
        box = [3, 4, 2, 2]
        image_width = 14
        box = td.lib.data_utils.bbox_xywh_hflip(box, image_width)
        assert box == [9, 4, 2, 2]

if __name__ == '__main__':
    unittest.main()
