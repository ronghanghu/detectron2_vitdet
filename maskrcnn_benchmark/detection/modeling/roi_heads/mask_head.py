import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d
from maskrcnn_benchmark.layers import cat


def project_masks_on_boxes(segmentation_masks, proposals, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(segmentation_masks, proposals)
    # TODO put the proposals on the CPU, as the representation for the
    # masks is not efficient GPU-wise (possibly several small tensors for
    # representing a single instance mask)
    proposals = proposals.bbox.to(torch.device("cpu"))
    # TODO use RoIAlign.
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation,
        # instead of the list representation that was used
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.convert(mode="mask")
        masks.append(mask)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


def maskrcnn_loss(proposals, mask_logits, matched_targets, discretization_size):
    """
    Arguments:
        proposals (list[BoxList]): all the foreground proposals
        mask_logits (Tensor)
        matched_targets (list[BoxList]): one-to-one corresponds to the proposals.

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """
    labels = []
    mask_targets = []
    for proposals_per_image, matched_targets_per_image in zip(proposals, matched_targets):
        labels_per_image = matched_targets_per_image.get_field("labels").to(dtype=torch.int64)
        segmentation_masks = matched_targets_per_image.get_field("masks")
        masks_per_image = project_masks_on_boxes(
            segmentation_masks, proposals_per_image, discretization_size
        )

        labels.append(labels_per_image)
        mask_targets.append(masks_per_image)

    labels = cat(labels, dim=0)
    mask_targets = cat(mask_targets, dim=0)

    positive_inds = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[positive_inds]

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits[positive_inds, labels_pos], mask_targets
    )
    return mask_loss


def maskrcnn_inference(mask_logits, boxes):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and add it to the boxlist as an extra "mask" field.

    Args:
        mask_logits (Tensor): NCHW tensor.
        boxes (list[BoxList]): boxes with "labels" field.

    Returns:
        None. boxes will contain an extra "mask" field.
        The field contain the mask in the CNN output size.

        It does not make sense to resize to image size here, because that is a quantization step, and
        should be done by the caller of the entire model who knows the true original size of the image.
    """
    mask_prob = mask_logits.sigmoid()

    # select masks coresponding to the predicted classes
    num_masks = mask_logits.shape[0]
    labels = [bbox.get_field("labels") for bbox in boxes]
    labels = torch.cat(labels)
    index = torch.arange(num_masks, device=labels.device)
    mask_prob = mask_prob[index, labels][:, None]

    boxes_per_image = [len(box) for box in boxes]
    mask_prob = mask_prob.split(boxes_per_image, dim=0)

    for prob, box in zip(mask_prob, boxes):
        box.add_field("mask", prob)


class MaskRCNNConvUpsampleHead(nn.Module):
    def __init__(self, num_convs, num_classes, input_channels, feature_channels):
        super(MaskRCNNConvUpsampleHead, self).__init__()
        self.blocks = []

        for k in range(num_convs):
            layer = Conv2d(
                input_channels if k == 0 else feature_channels,
                feature_channels,
                3,
                stride=1,
                padding=1,
            )
            self.add_module("mask_fcn{}".format(k + 1), layer)
            self.blocks.append(layer)

        self.deconv = ConvTranspose2d(
            feature_channels if num_convs > 0 else input_channels, feature_channels, 2, 2, 0
        )
        self.predictor = Conv2d(feature_channels, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        for layer in self.blocks:
            x = F.relu(layer(x))
        x = F.relu(self.deconv(x))
        return self.predictor(x)


def make_mask_head(cfg, input_size):
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
