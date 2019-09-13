import logging
import unittest
import torch

from detectron2.config import get_cfg
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator, RRPNAnchorGenerator

logger = logging.getLogger(__name__)


class TestAnchorGenerator(unittest.TestCase):
    def test_default_anchor_generator(self):
        cfg = get_cfg()
        cfg.MODEL.BACKBONE.COMPUTED_OUT_FEATURE_STRIDES = (("stage3", 4),)
        cfg.MODEL.RPN.IN_FEATURES = ["stage3"]
        cfg.MODEL.RPN.ANCHOR_SIZES = [[32, 64]]
        cfg.MODEL.RPN.ANCHOR_ASPECT_RATIOS = [[0.25, 1, 4]]
        feature_strides = dict(cfg.MODEL.BACKBONE.COMPUTED_OUT_FEATURE_STRIDES)

        anchor_generator = DefaultAnchorGenerator(
            cfg.MODEL.RPN.ANCHOR_SIZES,
            cfg.MODEL.RPN.ANCHOR_ASPECT_RATIOS,
            [feature_strides[f] for f in cfg.MODEL.RPN.IN_FEATURES],
        )

        num_images = 2
        # It's possible to infer strides from image size instead of config in the future.
        # For now, only len(images) is used in anchor_generator,
        # therefore here we just use 1 as a placeholder to represent the images.
        images = [1 for i in range(num_images)]
        # only the last two dimensions of features matter here
        features = {"stage3": torch.rand(14, 96, 1, 2)}
        anchors = anchor_generator(images, [features[f] for f in cfg.MODEL.RPN.IN_FEATURES])
        expected_anchor_tensor = torch.tensor(
            [
                [-32.0, -8.0, 32.0, 8.0],
                [-16.0, -16.0, 16.0, 16.0],
                [-8.0, -32.0, 8.0, 32.0],
                [-64.0, -16.0, 64.0, 16.0],
                [-32.0, -32.0, 32.0, 32.0],
                [-16.0, -64.0, 16.0, 64.0],
                [-28.0, -8.0, 36.0, 8.0],  # -28.0 == -32.0 + STRIDE (4)
                [-12.0, -16.0, 20.0, 16.0],
                [-4.0, -32.0, 12.0, 32.0],
                [-60.0, -16.0, 68.0, 16.0],
                [-28.0, -32.0, 36.0, 32.0],
                [-12.0, -64.0, 20.0, 64.0],
            ]
        )

        for i in range(num_images):
            assert torch.allclose(anchors[i][0].tensor, expected_anchor_tensor)

    def test_rrpn_anchor_generator(self):
        cfg = get_cfg()
        cfg.MODEL.BACKBONE.COMPUTED_OUT_FEATURE_STRIDES = (("stage3", 4),)
        cfg.MODEL.RPN.IN_FEATURES = ["stage3"]
        cfg.MODEL.RPN.ANCHOR_SIZES = [[32, 64]]
        cfg.MODEL.RPN.ANCHOR_ASPECT_RATIOS = [[0.25, 1, 4]]
        cfg.MODEL.RPN.ANCHOR_ANGLES = [[0, 45]]
        feature_strides = dict(cfg.MODEL.BACKBONE.COMPUTED_OUT_FEATURE_STRIDES)
        anchor_generator = RRPNAnchorGenerator(
            cfg.MODEL.RPN.ANCHOR_SIZES,
            cfg.MODEL.RPN.ANCHOR_ASPECT_RATIOS,
            cfg.MODEL.RPN.ANCHOR_ANGLES,
            [feature_strides[f] for f in cfg.MODEL.RPN.IN_FEATURES],
        )

        num_images = 2
        # It's possible to infer strides from image size instead of config in the future.
        # For now, only len(images) is used in anchor_generator,
        # therefore here we just use 1 as a placeholder to represent the images.
        images = [1 for i in range(num_images)]
        # only the last two dimensions of features matter here
        features = {"stage3": torch.rand(14, 96, 1, 2)}
        anchors = anchor_generator(images, [features[f] for f in cfg.MODEL.RPN.IN_FEATURES])
        expected_anchor_tensor = torch.tensor(
            [
                [0.0, 0.0, 64.0, 16.0, 0.0],
                [0.0, 0.0, 64.0, 16.0, 45.0],
                [0.0, 0.0, 32.0, 32.0, 0.0],
                [0.0, 0.0, 32.0, 32.0, 45.0],
                [0.0, 0.0, 16.0, 64.0, 0.0],
                [0.0, 0.0, 16.0, 64.0, 45.0],
                [0.0, 0.0, 128.0, 32.0, 0.0],
                [0.0, 0.0, 128.0, 32.0, 45.0],
                [0.0, 0.0, 64.0, 64.0, 0.0],
                [0.0, 0.0, 64.0, 64.0, 45.0],
                [0.0, 0.0, 32.0, 128.0, 0.0],
                [0.0, 0.0, 32.0, 128.0, 45.0],
                [4.0, 0.0, 64.0, 16.0, 0.0],  # 4.0 == 0.0 + STRIDE (4)
                [4.0, 0.0, 64.0, 16.0, 45.0],
                [4.0, 0.0, 32.0, 32.0, 0.0],
                [4.0, 0.0, 32.0, 32.0, 45.0],
                [4.0, 0.0, 16.0, 64.0, 0.0],
                [4.0, 0.0, 16.0, 64.0, 45.0],
                [4.0, 0.0, 128.0, 32.0, 0.0],
                [4.0, 0.0, 128.0, 32.0, 45.0],
                [4.0, 0.0, 64.0, 64.0, 0.0],
                [4.0, 0.0, 64.0, 64.0, 45.0],
                [4.0, 0.0, 32.0, 128.0, 0.0],
                [4.0, 0.0, 32.0, 128.0, 45.0],
            ]
        )

        for i in range(num_images):
            assert torch.allclose(anchors[i][0].tensor, expected_anchor_tensor)


if __name__ == "__main__":
    unittest.main()
