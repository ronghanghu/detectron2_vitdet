import logging
import numpy as np
import unittest

# uncomment for running in tests folder
# import sys
# sys.path.append("..")
from mosaic_augmentation import MosaicTransform

logger = logging.getLogger(__name__)


class TestMosaicTransform(unittest.TestCase):
    def test_image(self):
        img1 = np.zeros((10, 10, 4))
        img2 = np.ones((10, 10, 4))
        img3 = np.ones((10, 10, 4)) * 2
        img4 = np.ones((10, 10, 4)) * 3

        # no scaling
        T = MosaicTransform(20, 20, 10, 10, [10, 10, 10, 10], [10, 10, 10, 10])
        img = T.apply_image(img1, img2, img3, img4)

        assert (img[:10, :10] == 0).all(), "Top left quadrant is 0"
        assert (img[:10, 10:] == 1).all(), "Top right quadrant is 1"
        assert (img[10:, :10] == 2).all(), "Bottom left quadrant is 2"
        assert (img[10:, 10:] == 3).all(), "Bottom right quadrant is 3"

        # scale down
        T = MosaicTransform(10, 10, 5, 5, [10, 10, 10, 10], [10, 10, 10, 10])
        img = T.apply_image(img1, img2, img3, img4)

        assert (img[:5, :5] == 0).all(), "Top left quadrant is 0"
        assert (img[:5, 5:] == 1).all(), "Top right quadrant is 1"
        assert (img[5:, :5] == 2).all(), "Bottom left quadrant is 2"
        assert (img[5:, 5:] == 3).all(), "Bottom right quadrant is 3"

        # scale up
        T = MosaicTransform(40, 40, 20, 20, [10, 10, 10, 10], [10, 10, 10, 10])
        img = T.apply_image(img1, img2, img3, img4)

        assert (img[:20, :20] == 0).all(), "Top left quadrant is 0"
        assert (img[:20, 20:] == 1).all(), "Top right quadrant is 1"
        assert (img[20:, :20] == 2).all(), "Bottom left quadrant is 2"
        assert (img[20:, 20:] == 3).all(), "Bottom right quadrant is 3"

        # scale differently
        T = MosaicTransform(30, 30, 10, 10, [10, 10, 10, 10], [10, 10, 10, 10])
        img = T.apply_image(img1, img2, img3, img4)

        assert (img[:10, :10] == 0).all(), "Top left quadrant is 0"
        assert (img[:10, 10:] == 1).all(), "Top right quadrant is 1"
        assert (img[10:, :10] == 2).all(), "Bottom left quadrant is 2"
        assert (img[10:, 10:] == 3).all(), "Bottom right quadrant is 3"

    def test_coord(self):
        # original coordinates may change, so we need to reinitialize

        # no scale
        c1 = np.array([[2, 4]], dtype=float)
        c2 = np.array([[2, 6]], dtype=float)
        c3 = np.array([[2, 8]], dtype=float)
        c4 = np.array([[4, 6]], dtype=float)

        T = MosaicTransform(20, 20, 10, 10, [10, 10, 10, 10], [10, 10, 10, 10])
        c = T.apply_coords(c1, c2, c3, c4)
        assert (c == np.array([[2, 4], [12, 6], [2, 18], [14, 16]])).all(), c

        # scale down
        c1 = np.array([[2, 4]], dtype=float)
        c2 = np.array([[2, 6]], dtype=float)
        c3 = np.array([[2, 8]], dtype=float)
        c4 = np.array([[4, 6]], dtype=float)

        T = MosaicTransform(10, 10, 5, 5, [10, 10, 10, 10], [10, 10, 10, 10])
        c = T.apply_coords(c1, c2, c3, c4)
        assert (c == np.array([[1, 2], [6, 3], [1, 9], [7, 8]])).all(), c

        # scale up
        c1 = np.array([[2, 4]], dtype=float)
        c2 = np.array([[2, 6]], dtype=float)
        c3 = np.array([[2, 8]], dtype=float)
        c4 = np.array([[4, 6]], dtype=float)

        T = MosaicTransform(40, 40, 20, 20, [10, 10, 10, 10], [10, 10, 10, 10])
        c = T.apply_coords(c1, c2, c3, c4)
        assert (c == np.array([[4, 8], [24, 12], [4, 36], [28, 32]])).all(), c

        # scale differently
        c1 = np.array([[2, 4]], dtype=float)
        c2 = np.array([[2, 6]], dtype=float)
        c3 = np.array([[2, 8]], dtype=float)
        c4 = np.array([[4, 6]], dtype=float)

        T = MosaicTransform(40, 40, 10, 10, [10, 10, 10, 10], [10, 10, 10, 10])
        c = T.apply_coords(c1, c2, c3, c4)
        assert (c == np.array([[2, 4], [16, 6], [2, 34], [22, 28]])).all(), c

    def test_boxes(self):
        # original boxes may chance, so we must reinitialize

        # no scale
        b1 = np.array([[2, 4, 4, 6]], dtype=float)
        b2 = np.array([[2, 6, 4, 8]], dtype=float)
        b3 = np.array([[2, 8, 4, 8]], dtype=float)
        b4 = np.array([[4, 6, 6, 8]], dtype=float)

        T = MosaicTransform(20, 20, 10, 10, [10, 10, 10, 10], [10, 10, 10, 10])
        b = T.apply_box(b1, b2, b3, b4)
        assert (
            b == np.array([[2, 4, 4, 6], [12, 6, 14, 8], [2, 18, 4, 18], [14, 16, 16, 18]])
        ).all(), b

        # scale down
        b1 = np.array([[2, 4, 4, 6]], dtype=float)
        b2 = np.array([[2, 6, 4, 8]], dtype=float)
        b3 = np.array([[2, 8, 4, 8]], dtype=float)
        b4 = np.array([[4, 6, 6, 8]], dtype=float)

        T = MosaicTransform(10, 10, 5, 5, [10, 10, 10, 10], [10, 10, 10, 10])
        b = T.apply_box(b1, b2, b3, b4)
        assert (b == np.array([[1, 2, 2, 3], [6, 3, 7, 4], [1, 9, 2, 9], [7, 8, 8, 9]])).all(), b

        # scale up
        b1 = np.array([[2, 4, 4, 6]], dtype=float)
        b2 = np.array([[2, 6, 4, 8]], dtype=float)
        b3 = np.array([[2, 8, 4, 8]], dtype=float)
        b4 = np.array([[4, 6, 6, 8]], dtype=float)

        T = MosaicTransform(40, 40, 20, 20, [10, 10, 10, 10], [10, 10, 10, 10])
        b = T.apply_box(b1, b2, b3, b4)
        assert (
            b == np.array([[4, 8, 8, 12], [24, 12, 28, 16], [4, 36, 8, 36], [28, 32, 32, 36]])
        ).all(), b

        # scale differently
        b1 = np.array([[2, 4, 4, 6]], dtype=float)
        b2 = np.array([[2, 6, 4, 8]], dtype=float)
        b3 = np.array([[2, 8, 4, 8]], dtype=float)
        b4 = np.array([[4, 6, 6, 8]], dtype=float)

        T = MosaicTransform(40, 40, 10, 10, [10, 10, 10, 10], [10, 10, 10, 10])
        b = T.apply_box(b1, b2, b3, b4)
        assert (
            b == np.array([[2, 4, 4, 6], [16, 6, 22, 8], [2, 34, 4, 34], [22, 28, 28, 34]])
        ).all(), b

    def test_polygons(self):
        # original polygons may chance, so we must reinitialize

        # no scale
        p1 = [np.array([[2, 4], [4, 6], [2, 6]], dtype=float)]
        p2 = [np.array([[2, 6], [4, 8], [2, 8]], dtype=float)]
        p3 = [np.array([[2, 8], [4, 8], [2, 8]], dtype=float)]
        p4 = [np.array([[4, 6], [6, 8], [4, 8]], dtype=float)]

        T = MosaicTransform(20, 20, 10, 10, [10, 10, 10, 10], [10, 10, 10, 10])
        p = T.apply_polygons(p1, p2, p3, p4)
        true_p = [
            np.array([[2, 4], [4, 6], [2, 6]], dtype=float),
            np.array([[12, 6], [14, 8], [12, 8]], dtype=float),
            np.array([[2, 18], [4, 18], [2, 18]], dtype=float),
            np.array([[14, 16], [16, 18], [14, 18]], dtype=float),
        ]
        assert all([(p[i] == true_p[i]).all() for i in range(len(p))]), p

        # scale down
        p1 = [np.array([[2, 4], [4, 6], [2, 6]], dtype=float)]
        p2 = [np.array([[2, 6], [4, 8], [2, 8]], dtype=float)]
        p3 = [np.array([[2, 8], [4, 8], [2, 8]], dtype=float)]
        p4 = [np.array([[4, 6], [6, 8], [4, 8]], dtype=float)]

        T = MosaicTransform(10, 10, 5, 5, [10, 10, 10, 10], [10, 10, 10, 10])
        p = T.apply_polygons(p1, p2, p3, p4)
        true_p = [
            np.array([[1, 2], [2, 3], [1, 3]], dtype=float),
            np.array([[6, 3], [7, 4], [6, 4]], dtype=float),
            np.array([[1, 9], [2, 9], [1, 9]], dtype=float),
            np.array([[7, 8], [8, 9], [7, 9]], dtype=float),
        ]
        assert all([(p[i] == true_p[i]).all() for i in range(len(p))]), p

        # scale up
        p1 = [np.array([[2, 4], [4, 6], [2, 6]], dtype=float)]
        p2 = [np.array([[2, 6], [4, 8], [2, 8]], dtype=float)]
        p3 = [np.array([[2, 8], [4, 8], [2, 8]], dtype=float)]
        p4 = [np.array([[4, 6], [6, 8], [4, 8]], dtype=float)]

        T = MosaicTransform(40, 40, 20, 20, [10, 10, 10, 10], [10, 10, 10, 10])
        p = T.apply_polygons(p1, p2, p3, p4)
        true_p = [
            np.array([[4, 8], [8, 12], [4, 12]], dtype=float),
            np.array([[24, 12], [28, 16], [24, 16]], dtype=float),
            np.array([[4, 36], [8, 36], [4, 36]], dtype=float),
            np.array([[28, 32], [32, 36], [28, 36]], dtype=float),
        ]
        assert all([(p[i] == true_p[i]).all() for i in range(len(p))]), p

        # scale differently
        p1 = [np.array([[2, 4], [4, 6], [2, 6]], dtype=float)]
        p2 = [np.array([[2, 6], [4, 8], [2, 8]], dtype=float)]
        p3 = [np.array([[2, 8], [4, 8], [2, 8]], dtype=float)]
        p4 = [np.array([[4, 6], [6, 8], [4, 8]], dtype=float)]

        T = MosaicTransform(40, 40, 10, 10, [10, 10, 10, 10], [10, 10, 10, 10])
        p = T.apply_polygons(p1, p2, p3, p4)
        true_p = [
            np.array([[2, 4], [4, 6], [2, 6]], dtype=float),
            np.array([[16, 6], [22, 8], [16, 8]], dtype=float),
            np.array([[2, 34], [4, 34], [2, 34]], dtype=float),
            np.array([[22, 28], [28, 34], [22, 34]], dtype=float),
        ]
        assert all([(p[i] == true_p[i]).all() for i in range(len(p))]), p


if __name__ == "__main__":
    unittest.main()
