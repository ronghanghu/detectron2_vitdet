import torch_detectron as td
import unittest

class Tester(unittest.TestCase):

    def test_bbox_xywh_hflip(self):
        box = [3, 4, 2, 2]
        image_width = 14
        box = td.lib.data_utils.bbox_xywh_hflip(box, image_width)
        assert box == [9, 4, 2, 2]

    def test_bbox_xyxy_hflip(self):
        box = [2, 6, 10, 13]
        image_width = 18
        box = td.lib.data_utils.bbox_xyxy_hflip(box, image_width)
        assert box == [7, 6, 15, 13]

if __name__ == '__main__':
    unittest.main()
