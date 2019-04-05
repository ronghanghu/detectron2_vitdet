import torch
from torch import nn
from torch.nn import functional as F

from detectron2.structures import ImageList

from .backbone import build_backbone
from .model_builder import META_ARCH_REGISTRY
from .sem_seg_heads import build_sem_seg_head


@META_ARCH_REGISTRY.register()
class SemanticSegmentator(nn.Module):
    """
    Main class for semantic segmentation architectures.
    """

    def __init__(self, cfg):
        super(SemanticSegmentator, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.sem_seg_head = build_sem_seg_head(cfg)

        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DetectionTransform` .
                Each item in the list contains the inputs for one image.

        For now, each item in the list is a dict that contains:
            image: Tensor, image in (C, H, W) format.
            sem_seg_gt: semantic segmentation ground truth
            Other information that's included in the original dicts, such as:
                "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        """

        def make_image_list(batched_inputs, key, size_divisibility, pad_value=0):
            images = [x[key] for x in batched_inputs]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
            images = images.to(self.device)
            return images

        images = make_image_list(batched_inputs, "image", self.backbone.size_divisibility)

        if "sem_seg_gt" in batched_inputs[0]:
            targets = make_image_list(
                batched_inputs,
                "sem_seg_gt",
                self.backbone.size_divisibility,
                self.sem_seg_head.ignore_value,
            ).tensor
        else:
            targets = None

        features = self.backbone(images.tensor)
        results, losses = self.sem_seg_head(features, targets)

        if self.training:
            return losses

        processed_results = []
        for result, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            r = self.postprocess(result, image_size, height, width)
            processed_results.append(r)
        return processed_results

    def postprocess(self, result, img_size, output_height, output_width):
        """
        Postprocess the output semantic segmentation logit predictions. Return semantic segmentation
        labels predicted for each pixel in the original resolution.

        The input images are often resized when entering semantic segmentator. Moreover, in same
        cases, they also padded inside segmentator to be divisible by maximum network stride.
        As a result, we often need the predictions of the segmentator in a different
        resolution from its inputs.

        After resizing the logits to the desired resolutions, argmax is applied to return semnantic
        segmentation classes predicted for each pixel.

        Args:
            result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
                where C is the number of classes, and H, W are the height and width of the
                prediction.
            img_size (tuple): image size that segmentator is taking as input.
            output_height, output_width: the desired output resolution.

        Returns:
            semantic segmenation prediction (Tensor): A tensor of the shape
                (output_height, output_width) that contains per-pixel semantic segementation
                prediction.
        """
        result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
        result = F.interpolate(result, size=(output_height, output_width), mode="bilinear")[0]
        return result.argmax(dim=0)
