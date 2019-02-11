import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d, ConvTranspose2d, cat


def get_mask_ground_truth(masks_gt, box_list_pred, mask_side_len):
    """
    Given original ground-truth masks for an image, construct new ground-truth masks
    for a set of predicted boxes in that image. For each box, its ground-truth mask is
    constructed by clipping the corresponding original mask to the box, scaling it to
    the desired size and then rasterizing it into a tensor.

    Args:
        masks_gt (SegmentationList): A SegmentationList storing ground-truth masks for an image.
        box_list_pred (BoxList): A BoxList storing the predicted boxes for which ground-truth
            masks will be constructed.
        mask_side_len (int): The side length of the rasterized ground-truth masks.

    Returns:
        mask_logits_gt (Tensor): A tensor of shape (Bimg, mask_side_len, mask_side_len), where
            Bimg is the number of predicted boxes for this image. mask_logits_gt[i] stores the
            ground-truth for the predicted mask logits for the i-th predicted box.
    """
    device = box_list_pred.bbox.device
    box_list_pred = box_list_pred.convert("xyxy")
    assert masks_gt.size == box_list_pred.size, "{}, {}".format(masks_gt, box_list_pred)
    # Put box_list_pred on the CPU, as the representation for masks is not efficient
    # GPU-wise (possibly several small tensors for representing a single instance mask)
    box_list_pred = box_list_pred.bbox.to(torch.device("cpu"))
    mask_shape = (mask_side_len, mask_side_len)

    mask_logits_gt = []
    for mask_gt, box_pred in zip(masks_gt, box_list_pred):
        """
        mask_gt: a :class:`Polygons` instance
        box_pred: a tensor of shape (4,)
        """
        # Clip the ground-truth mask to the predicted box
        clipped_mask_gt = mask_gt.crop(box_pred)
        # Scale to the target mask shape
        scaled_clipped_mask_gt = clipped_mask_gt.resize(mask_shape)
        # Rasterize the scaled, clipped ground-truth mask
        rasterized_mask_gt = scaled_clipped_mask_gt.convert(mode="mask")
        mask_logits_gt.append(rasterized_mask_gt)
    if len(mask_logits_gt) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(mask_logits_gt, dim=0).to(device, dtype=torch.float32)


def mask_rcnn_loss(mask_logits_pred, box_lists_pred, box_lists_gt, mask_side_len):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        mask_logits_pred (Tensor): A tensor of shape (B, C, Hmask, Wmask), where B is the
            total number of predicted masks in all images, C is the number of classes,
            and Hmask, Wmask are the height and width of the mask predictions. The values
            are logits.
        box_lists_pred (list[BoxList]): A list of N BoxLists, where N is the number of images
            in the batch. These boxes are predictions from the model that are in 1:1
            correspondence with the mask_logits_pred.
        box_lists_gt (list[BoxList]): Ground-truth boxes in one-to-one corresponds with
            box_lists_pred.
        mask_side_len (int): The side length of the rasterized ground-truth masks.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    class_gt = []
    mask_logits_gt = []
    for box_list_pred_per_image, box_list_gt_per_image in zip(box_lists_pred, box_lists_gt):
        class_gt_per_image = box_list_gt_per_image.get_field("labels").to(dtype=torch.int64)
        masks_gt = box_list_gt_per_image.get_field("masks")
        mask_logits_gt_per_image = get_mask_ground_truth(
            masks_gt, box_list_pred_per_image, mask_side_len
        )
        class_gt.append(class_gt_per_image)
        mask_logits_gt.append(mask_logits_gt_per_image)

    class_gt = cat(class_gt, dim=0)
    mask_logits_gt = cat(mask_logits_gt, dim=0)

    # Masks should not be predicted for boxes that were not matched to a gt box.
    assert torch.all(class_gt > 0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_logits_gt.numel() == 0:
        return mask_logits_pred.sum() * 0

    indices = torch.arange(len(class_gt))
    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits_pred[indices, class_gt], mask_logits_gt, reduction="mean"
    )
    return mask_loss


def mask_rcnn_inference(mask_logits_pred, box_lists_pred):
    """
    Convert mask_logits_pred to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in box_lists_pred. For each
    predicted box, the mask of the same class is attached to the box by adding a
    new "mask" field to box_lists_pred.

    Args:
        mask_logits_pred (Tensor): A tensor of shape (B, C, Hmask, Wmask), where B is the
            total number of predicted masks in all images, C is the number of classes,
            and Hmask, Wmask are the height and width of the mask predictions. The values
            are logits.
        box_lists_pred (list[BoxList]): A list of N BoxLists, where N is the number of images
            in the batch. Each BoxList has a "labels" field.

    Returns:
        None. box_lists_pred will contain an extra "mask" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    mask_probs_pred = mask_logits_pred.sigmoid()

    # Select masks coresponding to the predicted classes
    num_masks = mask_logits_pred.shape[0]
    class_pred = torch.cat([box_list.get_field("labels") for box_list in box_lists_pred])
    indices = torch.arange(num_masks, device=class_pred.device)
    mask_probs_pred = mask_probs_pred[indices, class_pred][:, None]

    num_boxes_per_image = [len(box_list) for box_list in box_lists_pred]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, box in zip(mask_probs_pred, box_lists_pred):
        box.add_field("mask", prob)


class MaskRCNNConvUpsampleHead(nn.Module):
    def __init__(self, num_convs, num_classes, input_channels, feature_channels):
        super(MaskRCNNConvUpsampleHead, self).__init__()
        self.convs = []

        for k in range(num_convs):
            layer = Conv2d(
                input_channels if k == 0 else feature_channels,
                feature_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.add_module("mask_fcn{}".format(k + 1), layer)
            self.convs.append(layer)

        self.deconv = ConvTranspose2d(
            feature_channels if num_convs > 0 else input_channels,
            feature_channels,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.predictor = Conv2d(feature_channels, num_classes, kernel_size=1, stride=1, padding=0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        for conv in self.convs:
            x = F.relu(conv(x))
        x = F.relu(self.deconv(x))
        return self.predictor(x)


# TODO: use registration
def build_mask_head(cfg, input_size):
    """
    input_size: int (channels) or tuple (channels, height, width)
    """
    head = cfg.MODEL.ROI_MASK_HEAD.NAME
    num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
    if not isinstance(input_size, int):
        input_size = input_size[0]
    if head == "MaskRCNN4ConvUpsampleHead":
        return MaskRCNNConvUpsampleHead(
            4, num_classes, input_size, cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        )
    if head == "MaskRCNNUpsampleHead":
        return MaskRCNNConvUpsampleHead(
            0, num_classes, input_size, cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        )
    raise ValueError("Unknown head {}".format(head))
