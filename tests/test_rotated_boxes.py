from __future__ import absolute_import, division, print_function, unicode_literals
import math
import unittest
import torch
from borc.common.benchmark import benchmark

from detectron2.layers.rotated_boxes import pairwise_iou_rotated


class TestRotatedBoxes(unittest.TestCase):
    def test_iou_0_dim_cpu(self):
        boxes1 = torch.rand(0, 5, dtype=torch.float32)
        boxes2 = torch.rand(10, 5, dtype=torch.float32)
        expected_ious = torch.zeros(0, 10, dtype=torch.float32)
        ious = pairwise_iou_rotated(boxes1, boxes2)
        assert torch.allclose(ious, expected_ious)

        boxes1 = torch.rand(10, 5, dtype=torch.float32)
        boxes2 = torch.rand(0, 5, dtype=torch.float32)
        expected_ious = torch.zeros(10, 0, dtype=torch.float32)
        ious = pairwise_iou_rotated(boxes1, boxes2)
        assert torch.allclose(ious, expected_ious)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_iou_0_dim_cuda(self):
        boxes1 = torch.rand(0, 5, dtype=torch.float32)
        boxes2 = torch.rand(10, 5, dtype=torch.float32)
        expected_ious = torch.zeros(0, 10, dtype=torch.float32)
        ious_cuda = pairwise_iou_rotated(boxes1.cuda(), boxes2.cuda())
        assert torch.allclose(ious_cuda.cpu(), expected_ious)

        boxes1 = torch.rand(10, 5, dtype=torch.float32)
        boxes2 = torch.rand(0, 5, dtype=torch.float32)
        expected_ious = torch.zeros(10, 0, dtype=torch.float32)
        ious_cuda = pairwise_iou_rotated(boxes1.cuda(), boxes2.cuda())
        assert torch.allclose(ious_cuda.cpu(), expected_ious)

    def test_iou_half_overlap_cpu(self):
        boxes1 = torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.0]], dtype=torch.float32)
        boxes2 = torch.tensor([[0.25, 0.5, 0.5, 1.0, 0.0]], dtype=torch.float32)
        expected_ious = torch.tensor([[0.5]], dtype=torch.float32)
        ious = pairwise_iou_rotated(boxes1, boxes2)
        assert torch.allclose(ious, expected_ious)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_iou_half_overlap_cuda(self):
        boxes1 = torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.0]], dtype=torch.float32)
        boxes2 = torch.tensor([[0.25, 0.5, 0.5, 1.0, 0.0]], dtype=torch.float32)
        expected_ious = torch.tensor([[0.5]], dtype=torch.float32)
        ious_cuda = pairwise_iou_rotated(boxes1.cuda(), boxes2.cuda())
        assert torch.allclose(ious_cuda.cpu(), expected_ious)

    def test_iou_0_degree_cpu(self):
        boxes1 = torch.tensor(
            [[0.5, 0.5, 1.0, 1.0, 0.0], [0.5, 0.5, 1.0, 1.0, 0.0]], dtype=torch.float32
        )
        boxes2 = torch.tensor(
            [
                [0.5, 0.5, 1.0, 1.0, 0.0],
                [0.25, 0.5, 0.5, 1.0, 0.0],
                [0.5, 0.25, 1.0, 0.5, 0.0],
                [0.25, 0.25, 0.5, 0.5, 0.0],
                [0.75, 0.75, 0.5, 0.5, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        expected_ious = torch.tensor(
            [
                [1.0, 0.5, 0.5, 0.25, 0.25, 0.25 / (2 - 0.25)],
                [1.0, 0.5, 0.5, 0.25, 0.25, 0.25 / (2 - 0.25)],
            ],
            dtype=torch.float32,
        )
        ious = pairwise_iou_rotated(boxes1, boxes2)
        assert torch.allclose(ious, expected_ious)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_iou_0_degree_cuda(self):
        boxes1 = torch.tensor(
            [[0.5, 0.5, 1.0, 1.0, 0.0], [0.5, 0.5, 1.0, 1.0, 0.0]], dtype=torch.float32
        )
        boxes2 = torch.tensor(
            [
                [0.5, 0.5, 1.0, 1.0, 0.0],
                [0.25, 0.5, 0.5, 1.0, 0.0],
                [0.5, 0.25, 1.0, 0.5, 0.0],
                [0.25, 0.25, 0.5, 0.5, 0.0],
                [0.75, 0.75, 0.5, 0.5, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        expected_ious = torch.tensor(
            [
                [1.0, 0.5, 0.5, 0.25, 0.25, 0.25 / (2 - 0.25)],
                [1.0, 0.5, 0.5, 0.25, 0.25, 0.25 / (2 - 0.25)],
            ],
            dtype=torch.float32,
        )
        ious_cuda = pairwise_iou_rotated(boxes1.cuda(), boxes2.cuda())
        assert torch.allclose(ious_cuda.cpu(), expected_ious)

    def test_iou_45_degrees_cpu(self):
        boxes1 = torch.tensor(
            [
                [1, 1, math.sqrt(2), math.sqrt(2), 45],
                [1, 1, 2 * math.sqrt(2), 2 * math.sqrt(2), -45],
            ],
            dtype=torch.float32,
        )
        boxes2 = torch.tensor([[1, 1, 2, 2, 0]], dtype=torch.float32)
        expected_ious = torch.tensor([[0.5], [0.5]], dtype=torch.float32)
        ious = pairwise_iou_rotated(boxes1, boxes2)
        assert torch.allclose(ious, expected_ious)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_iou_45_degrees_cuda(self):
        boxes1 = torch.tensor(
            [
                [1, 1, math.sqrt(2), math.sqrt(2), 45],
                [1, 1, 2 * math.sqrt(2), 2 * math.sqrt(2), -45],
            ],
            dtype=torch.float32,
        )
        boxes2 = torch.tensor([[1, 1, 2, 2, 0]], dtype=torch.float32)
        expected_ious = torch.tensor([[0.5], [0.5]], dtype=torch.float32)
        ious_cuda = pairwise_iou_rotated(boxes1.cuda(), boxes2.cuda())
        assert torch.allclose(ious_cuda.cpu(), expected_ious)

    def test_iou_perpendicular_cpu(self):
        boxes1 = torch.tensor([[5, 5, 10.0, 6, 55]], dtype=torch.float32)
        boxes2 = torch.tensor([[5, 5, 10.0, 6, -35]], dtype=torch.float32)
        iou = (6.0 * 6.0) / (6.0 * 6.0 + 4.0 * 6.0 + 4.0 * 6.0)
        expected_ious = torch.tensor([[iou]], dtype=torch.float32)
        ious = pairwise_iou_rotated(boxes1, boxes2)
        assert torch.allclose(ious, expected_ious)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_iou_perpendicular_cuda(self):
        boxes1 = torch.tensor([[5, 5, 10.0, 6, 55]], dtype=torch.float32)
        boxes2 = torch.tensor([[5, 5, 10.0, 6, -35]], dtype=torch.float32)
        iou = (6.0 * 6.0) / (6.0 * 6.0 + 4.0 * 6.0 + 4.0 * 6.0)
        expected_ious = torch.tensor([[iou]], dtype=torch.float32)
        ious_cuda = pairwise_iou_rotated(boxes1.cuda(), boxes2.cuda())
        assert torch.allclose(ious_cuda.cpu(), expected_ious)

    def test_iou_large_close_boxes_cpu(self):
        boxes1 = torch.tensor(
            [[299.500000, 417.370422, 600.000000, 364.259186, 27.1828]], dtype=torch.float32
        )
        boxes2 = torch.tensor(
            [[299.500000, 417.370422, 600.000000, 364.259155, 27.1828]], dtype=torch.float32
        )
        iou = 364.259155 / 364.259186
        expected_ious = torch.tensor([[iou]], dtype=torch.float32)
        ious = pairwise_iou_rotated(boxes1, boxes2)
        assert torch.allclose(ious, expected_ious)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_iou_large_close_boxes_cuda(self):
        boxes1 = torch.tensor(
            [[299.500000, 417.370422, 600.000000, 364.259186, 27.1828]], dtype=torch.float32
        )
        boxes2 = torch.tensor(
            [[299.500000, 417.370422, 600.000000, 364.259155, 27.1828]], dtype=torch.float32
        )
        iou = 364.259155 / 364.259186
        expected_ious = torch.tensor([[iou]], dtype=torch.float32)
        ious_cuda = pairwise_iou_rotated(boxes1.cuda(), boxes2.cuda())
        assert torch.allclose(ious_cuda.cpu(), expected_ious)

    def test_iou_precision_cpu(self):
        boxes1 = torch.tensor([[565, 565, 10, 10, 0]], dtype=torch.float32)
        boxes2 = torch.tensor([[565, 565, 10, 8.3, 0]], dtype=torch.float32)
        iou = 8.3 / 10.0
        expected_ious = torch.tensor([[iou]], dtype=torch.float32)
        ious = pairwise_iou_rotated(boxes1, boxes2)
        assert torch.allclose(ious, expected_ious)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_iou_precision_cuda(self):
        boxes1 = torch.tensor([[565, 565, 10, 10, 0]], dtype=torch.float32)
        boxes2 = torch.tensor([[565, 565, 10, 8.3, 0]], dtype=torch.float32)
        iou = 8.3 / 10.0
        expected_ious = torch.tensor([[iou]], dtype=torch.float32)
        ious_cuda = pairwise_iou_rotated(boxes1.cuda(), boxes2.cuda())
        assert torch.allclose(ious_cuda.cpu(), expected_ious)

    def test_iou_many_boxes_cpu(self):
        num_boxes1 = 100
        num_boxes2 = 200
        boxes1 = torch.stack(
            [
                torch.tensor([5 + 20 * i, 5 + 20 * i, 10, 10, 0], dtype=torch.float32)
                for i in range(num_boxes1)
            ]
        )
        boxes2 = torch.stack(
            [
                torch.tensor(
                    [5 + 20 * i, 5 + 20 * i, 10, 1 + 9 * i / num_boxes2, 0], dtype=torch.float32
                )
                for i in range(num_boxes2)
            ]
        )
        expected_ious = torch.zeros(num_boxes1, num_boxes2, dtype=torch.float32)
        for i in range(min(num_boxes1, num_boxes2)):
            expected_ious[i][i] = (1 + 9 * i / num_boxes2) / 10.0
        ious = pairwise_iou_rotated(boxes1, boxes2)
        assert torch.allclose(ious, expected_ious)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_iou_many_boxes_cuda(self):
        num_boxes1 = 100
        num_boxes2 = 200
        boxes1 = torch.stack(
            [
                torch.tensor([5 + 20 * i, 5 + 20 * i, 10, 10, 0], dtype=torch.float32)
                for i in range(num_boxes1)
            ]
        )
        boxes2 = torch.stack(
            [
                torch.tensor(
                    [5 + 20 * i, 5 + 20 * i, 10, 1 + 9 * i / num_boxes2, 0], dtype=torch.float32
                )
                for i in range(num_boxes2)
            ]
        )
        expected_ious = torch.zeros(num_boxes1, num_boxes2, dtype=torch.float32)
        for i in range(min(num_boxes1, num_boxes2)):
            expected_ious[i][i] = (1 + 9 * i / num_boxes2) / 10.0
        ious_cuda = pairwise_iou_rotated(boxes1.cuda(), boxes2.cuda())
        assert torch.allclose(ious_cuda.cpu(), expected_ious)

    def test_benchmark_cpu_cuda(self):
        num_boxes1 = 500
        num_boxes2 = 1000
        boxes1 = torch.stack(
            [
                torch.tensor([5 + 20 * i, 5 + 20 * i, 10, 10, 0], dtype=torch.float32)
                for i in range(num_boxes1)
            ]
        )
        boxes2 = torch.stack(
            [
                torch.tensor(
                    [5 + 20 * i, 5 + 20 * i, 10, 1 + 9 * i / num_boxes2, 0], dtype=torch.float32
                )
                for i in range(num_boxes2)
            ]
        )

        def func(dev, n=1):
            b1 = boxes1.to(device=dev)
            b2 = boxes2.to(device=dev)

            def bench():
                for _ in range(n):
                    pairwise_iou_rotated(b1, b2)
                if dev.type == "cuda":
                    torch.cuda.synchronize()

            return bench

        # only run it once per timed loop, since it's slow
        args = [{"dev": torch.device("cpu"), "n": 1}]
        if torch.cuda.is_available():
            args.append({"dev": torch.device("cuda"), "n": 10})

        benchmark(func, "rotated_iou", args, warmup_iters=3)


if __name__ == "__main__":
    unittest.main()
