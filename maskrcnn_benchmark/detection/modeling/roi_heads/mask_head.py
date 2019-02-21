import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d, ConvTranspose2d, cat, weight_init
from maskrcnn_benchmark.utils.events import get_event_storage


def get_mask_ground_truth(gt_masks, pred_boxes, mask_side_len):
    """
    Given original ground-truth masks for an image, construct new ground-truth masks
    for a set of predicted boxes in that image. For each box, its ground-truth mask is
    constructed by clipping the corresponding original mask to the box, scaling it to
    the desired size and then rasterizing it into a tensor.

    Args:
        gt_masks (SegmentationList): A SegmentationList storing ground-truth masks for an image.
        pred_boxes (Boxes): A Boxes storing the predicted boxes for which ground-truth
            masks will be constructed.
        mask_side_len (int): The side length of the rasterized ground-truth masks.

    Returns:
        gt_mask_logits (Tensor): A byte tensor of shape (Bimg, mask_side_len, mask_side_len), where
            Bimg is the number of predicted boxes for this image. gt_mask_logits[i] stores the
            ground-truth for the predicted mask logits for the i-th predicted box.
    """
    device = pred_boxes.device
    assert pred_boxes.mode == "xyxy"
    # Put pred_boxes on the CPU, as the representation for masks is not efficient
    # GPU-wise (possibly several small tensors for representing a single instance mask)
    pred_boxes = pred_boxes.to(torch.device("cpu"))
    mask_shape = (mask_side_len, mask_side_len)

    gt_mask_logits = []
    for gt_mask, pred_box in zip(gt_masks, pred_boxes):
        """
        gt_mask: a :class:`Polygons` instance
        pred_box: a tensor of shape (4,)
        """
        # Clip the ground-truth mask to the predicted box
        clipped_gt_mask = gt_mask.crop(pred_box)
        # Scale to the target mask shape
        scaled_clipped_gt_mask = clipped_gt_mask.resize(mask_shape)
        # Rasterize the scaled, clipped ground-truth mask
        rasterized_gt_mask = scaled_clipped_gt_mask.convert_to_mask()
        gt_mask_logits.append(rasterized_gt_mask)
    if len(gt_mask_logits) == 0:
        return torch.empty(0, dtype=torch.uint8, device=device)
    return torch.stack(gt_mask_logits, dim=0).to(device=device)


def mask_rcnn_loss(pred_mask_logits, instances, mask_side_len):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask), where B is the
            total number of predicted masks in all images, C is the number of classes,
            and Hmask, Wmask are the height and width of the mask predictions. The values
            are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        mask_side_len (int): The side length of the rasterized ground-truth masks.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    gt_classes = []
    gt_mask_logits = []
    for instances_per_image in instances:
        gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
        gt_masks = instances_per_image.gt_masks
        gt_mask_logits_per_image = get_mask_ground_truth(
            gt_masks, instances_per_image.proposal_boxes, mask_side_len
        )
        gt_classes.append(gt_classes_per_image)
        gt_mask_logits.append(gt_mask_logits_per_image)

    gt_classes = cat(gt_classes, dim=0)
    gt_mask_logits = cat(gt_mask_logits, dim=0)

    # Masks should not be predicted for boxes that were not matched to a gt box.
    assert torch.all(gt_classes > 0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if gt_mask_logits.numel() == 0:
        return pred_mask_logits.sum() * 0

    indices = torch.arange(len(gt_classes))
    pred_mask_logits = pred_mask_logits[indices, gt_classes]

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_accurate = (pred_mask_logits > 0.5) == gt_mask_logits
    mask_accuracy = mask_accurate.nonzero().size(0) / mask_accurate.numel()
    get_event_storage().put_scalar("mask_rcnn/accuracy", mask_accuracy)

    mask_loss = F.binary_cross_entropy_with_logits(
        pred_mask_logits, gt_mask_logits.to(dtype=torch.float32), reduction="mean"
    )
    return mask_loss


def mask_rcnn_inference(pred_mask_logits, pred_instances):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask), where B is the
            total number of predicted masks in all images, C is the number of classes,
            and Hmask, Wmask are the height and width of the mask predictions. The values
            are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    mask_probs_pred = pred_mask_logits.sigmoid()

    # Select masks coresponding to the predicted classes
    num_masks = pred_mask_logits.shape[0]
    class_pred = torch.cat([i.pred_classes for i in pred_instances])
    indices = torch.arange(num_masks, device=class_pred.device)
    mask_probs_pred = mask_probs_pred[indices, class_pred][:, None]

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob


class MaskRCNNConvUpsampleHead(nn.Module):
    def __init__(self, num_convs, num_classes, input_channels, feature_channels, norm):
        super(MaskRCNNConvUpsampleHead, self).__init__()
        self.conv_norm_relus = []

        for k in range(num_convs):
            norm_module = nn.GroupNorm(32, feature_channels) if norm == "GN" else None
            conv = Conv2d(
                input_channels if k == 0 else feature_channels,
                feature_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not norm,
                norm=norm_module,
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)

        self.deconv = ConvTranspose2d(
            feature_channels if num_convs > 0 else input_channels,
            feature_channels,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.predictor = Conv2d(feature_channels, num_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv, self.predictor]:
            weight_init.c2_msra_fill(layer)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        x = F.relu(self.deconv(x))
        return self.predictor(x)


# TODO: use registration
def build_mask_head(cfg, input_size):
    """
    input_size: int (channels) or tuple (channels, height, width)
    """
    head = cfg.MODEL.ROI_MASK_HEAD.NAME
    num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
    norm = cfg.MODEL.ROI_MASK_HEAD.NORM
    if not isinstance(input_size, int):
        input_size = input_size[0]
    if head == "MaskRCNN4ConvUpsampleHead":
        return MaskRCNNConvUpsampleHead(
            4, num_classes, input_size, cfg.MODEL.ROI_MASK_HEAD.CONV_DIM, norm=norm
        )
    if head == "MaskRCNNUpsampleHead":
        return MaskRCNNConvUpsampleHead(
            0, num_classes, input_size, cfg.MODEL.ROI_MASK_HEAD.CONV_DIM, norm=norm
        )
    raise ValueError("Unknown head {}".format(head))
