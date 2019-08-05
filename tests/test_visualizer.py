# -*- coding: utf-8 -*-
# File:

import numpy as np
import unittest

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer


class TestVisualizer(unittest.TestCase):
    def _random_data(self):
        H, W = 100, 100
        N = 10
        img = np.random.rand(H, W, 3) * 255
        boxxy = np.random.rand(N, 2) * (H // 2)
        boxes = np.concatenate((boxxy, boxxy + H // 2), axis=1)

        def _rand_poly():
            return np.random.rand(3, 2).flatten() * H

        polygons = [[_rand_poly() for _ in range(np.random.randint(1, 5))] for _ in range(N)]

        labels = [str(i) for i in range(N)]
        return img, boxes, labels, polygons

    @property
    def metadata(self):
        return MetadataCatalog.get("coco_2017_train")

    def test_overlay_instances(self):
        img, boxes, labels, polygons = self._random_data()

        v = Visualizer(img, self.metadata)
        output = v.overlay_instances(masks=polygons, boxes=boxes, labels=labels).get_image()
        self.assertEqual(output.shape, img.shape)

        # Test 2x scaling
        v = Visualizer(img, self.metadata, scale=2.0)
        output = v.overlay_instances(masks=polygons, boxes=boxes, labels=labels).get_image()
        self.assertEqual(output.shape[0], img.shape[0] * 2)

    def test_overlay_instances_no_boxes(self):
        img, boxes, labels, polygons = self._random_data()
        v = Visualizer(img, self.metadata)
        v.overlay_instances(masks=polygons, boxes=None, labels=labels).get_image()
