import unittest
import torch

from detectron2.detection.modeling.box_regression import Box2BoxTransform
from detectron2.structures.boxes import Boxes, pairwise_iou


def random_boxes(mean_box, stdev, N):
    return torch.rand(N, 4) * stdev + torch.tensor(mean_box, dtype=torch.float)


class TestBox2BoxTransform(unittest.TestCase):
    def test_reconstruction(self):
        weights = (5, 5, 10, 10)
        b2b_tfm = Box2BoxTransform(weights=weights)
        src_boxes = random_boxes([10, 10, 20, 20], 1, 10)
        dst_boxes = random_boxes([10, 10, 20, 20], 1, 10)
        deltas = b2b_tfm.get_deltas(src_boxes, dst_boxes)
        dst_boxes_reconstructed = b2b_tfm.apply_deltas(deltas, src_boxes)
        assert torch.allclose(dst_boxes, dst_boxes_reconstructed)

    def test_pairwise_iou(self):
        boxes1 = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])

        boxes2 = torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.5, 1.0],
                [0.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 0.5, 0.5],
                [0.5, 0.5, 1.0, 1.0],
                [0.5, 0.5, 1.5, 1.5],
            ]
        )

        expected_ious = torch.tensor(
            [
                [1.0, 0.5, 0.5, 0.25, 0.25, 0.25 / (2 - 0.25)],
                [1.0, 0.5, 0.5, 0.25, 0.25, 0.25 / (2 - 0.25)],
            ]
        )

        ious = pairwise_iou(Boxes(boxes1), Boxes(boxes2))

        assert torch.allclose(ious, expected_ious)


if __name__ == "__main__":
    unittest.main()
