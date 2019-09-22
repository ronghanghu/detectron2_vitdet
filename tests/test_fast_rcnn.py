import logging
import unittest
import torch

from detectron2.config import get_cfg
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.events import EventStorage

logger = logging.getLogger(__name__)


class FastRCNNTest(unittest.TestCase):
    def test_fast_rcnn(self):
        torch.manual_seed(132)
        cfg = get_cfg()
        box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

        box_head_output_size = 8
        num_classes = 5
        cls_agnostic_bbox_reg = False

        box_predictor = FastRCNNOutputLayers(
            box_head_output_size, num_classes, cls_agnostic_bbox_reg
        )
        feature_pooled = torch.rand(2, box_head_output_size)
        pred_class_logits, pred_proposal_deltas = box_predictor(feature_pooled)
        image_shape = (10, 10)
        proposal_boxes = torch.tensor([[0.8, 1.1, 3.2, 2.8], [2.3, 2.5, 7, 8]], dtype=torch.float32)
        gt_boxes = torch.tensor([[1, 1, 3, 3], [2, 2, 6, 6]], dtype=torch.float32)
        result = Instances(image_shape)
        result.proposal_boxes = Boxes(proposal_boxes)
        result.gt_boxes = Boxes(gt_boxes)
        result.gt_classes = torch.tensor([1, 2])
        proposals = []
        proposals.append(result)
        smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA

        outputs = FastRCNNOutputs(
            box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta
        )
        with EventStorage():  # capture events in a new storage to discard them
            losses = outputs.losses()

        expected_losses = {
            "loss_cls": torch.tensor(1.7951188087),
            "loss_box_reg": torch.tensor(4.0357131958),
        }
        for name in expected_losses.keys():
            assert torch.allclose(losses[name], expected_losses[name])


if __name__ == "__main__":
    unittest.main()
