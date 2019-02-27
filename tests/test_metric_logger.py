# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import unittest

from detectron2.utils.metric_logger import MetricLogger


class TestMetricLogger(unittest.TestCase):
    def test_update(self):
        meter = MetricLogger()
        for i in range(10):
            meter.update(metric=float(i))

        m = meter.meters["metric"]
        self.assertEqual(m.count, 10)
        self.assertEqual(m.total, 45)
        self.assertEqual(m.median, 4)
        self.assertEqual(m.avg, 4.5)

    def test_no_attr(self):
        meter = MetricLogger()
        meter.meters
        meter.delimiter

        def broken():
            meter.not_existent

        self.assertRaises(AttributeError, broken)


if __name__ == "__main__":
    unittest.main()
