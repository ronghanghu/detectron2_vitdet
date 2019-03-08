import copy
import torch
from torch import nn

from detectron2.structures import Instances
from detectron2.utils.registry import Registry

from .backbone import build_backbone
from .roi_heads.paste_mask import paste_masks_in_image
from .roi_heads.roi_heads import build_roi_heads
from .rpn.rpn import build_rpn

META_ARCH_REGISTRY = Registry("META_ARCH")


def build_detection_model(cfg):
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    return META_ARCH_REGISTRY.get(meta_arch)(cfg)


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
        self.roi_heads = build_roi_heads(cfg)

        self.to(self.device)

    def forward(self, data):
        """
        Arguments:
            data: a tuple, produced by :class:`DetectionBatchCollator`.

        For now, the data contains images, targets, and images_metadata
        images: ImageList
        targets: list[Instances]
        image_metadata: list[dict], metadata for each image that may be useful.
            Currently, it only needs:
                "height", "width" (int): the output resolution of the model.
                See :meth:`postprocess` for details.
        """

        images, targets, image_metadata = data
        images = images.to(self.device)
        if targets is not None:
            targets = [t.to(self.device) for t in targets]

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
        for results_per_image, metadata_per_image, image_size in zip(
            results, image_metadata, images.image_sizes
        ):
            height = metadata_per_image.get("height", image_size[0])
            width = metadata_per_image.get("width", image_size[1])
            r = self.postprocess(results_per_image, height, width)
            processed_results.append(r)
        return processed_results

    def postprocess(self, results, output_height, output_width):
        """
        Postprocess the output boxes.
        The input images are often resized when entering an object detector.
        As a result, we often need the outputs of the detector in a different
        resolution from its inputs.

        This function will postprocess the raw outputs of an R-CNN detector
        to produce outputs according to the desired output resolution.

        Args:
            results (Instances): the raw outputs from the detector.
                `results.image_size` contains the input image resolution the detector sees.
            output_height, output_width: the desired output resolution.

        Returns:
            Instances: the postprocessed output from the model, based on the output resolution
        """
        scale_x, scale_y = (
            output_width / results.image_size[1],
            output_height / results.image_size[0],
        )
        results = Instances((output_height, output_width), **copy.deepcopy(results.get_fields()))
        pred_boxes = results.pred_boxes
        pred_boxes.tensor[:, 0::2] *= scale_x
        pred_boxes.tensor[:, 1::2] *= scale_y
        pred_boxes.clip(results.image_size)
        results = results[pred_boxes.nonempty()]

        if results.has("pred_masks"):
            MASK_THRESHOLD = 0.5
            results.pred_masks = paste_masks_in_image(
                results.pred_masks,  # N, 1, M, M
                results.pred_boxes,
                results.image_size,
                threshold=MASK_THRESHOLD,
                padding=1,
            ).squeeze(1)

        if results.has("pred_keypoints"):
            results.pred_keypoints.tensor[:, :, 0] *= scale_x
            results.pred_keypoints.tensor[:, :, 1] *= scale_y

        return results
