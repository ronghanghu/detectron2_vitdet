import torch
from torch import nn

from detectron2.structures import ImageList

from .backbone import build_backbone
from .model_builder import META_ARCH_REGISTRY
from .postprocessing import detector_postprocess
from .roi_heads.roi_heads import build_roi_heads
from .rpn.rpn import build_rpn


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Supports boxes, masks and keypoints
    This is very similar to what we had before, the difference being that now
    we construct the modules in __init__, instead of passing them as arguments
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        if cfg.MODEL.RPN_ONLY:
            self.roi_heads = None
        else:
            self.roi_heads = build_roi_heads(cfg)

        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DetectionTransform` .
                Each item in the list contains the inputs for one image.

        For now, each item in the list is a dict that contains:
            image: Tensor, image in (C, H, W) format.
            targets: Instances
            Other information that's included in the original dicts, such as:
                "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "detector" whose value is a
                :class:`Instances`.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        if "targets" in batched_inputs[0]:
            targets = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            targets = None

        features = self.backbone(images.tensor)
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            results, detector_losses = self.roi_heads(images, features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads.
            results = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"detector": r})
        return processed_results
