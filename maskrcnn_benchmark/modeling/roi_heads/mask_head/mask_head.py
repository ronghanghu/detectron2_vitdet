import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d


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
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )
    # TODO put the proposals on the CPU, as the representation for the
    # masks is not efficient GPU-wise (possibly several small tensors for
    # representing a single instance mask)
    proposals = proposals.bbox.to(torch.device("cpu"))
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


class MaskRCNNConvUpsampleHead(nn.Module):
    def __init__(self, num_convs, num_classes, input_channels, feature_channels):
        super(MaskRCNNConvUpsampleHead, self).__init__()
        self.blocks = []

        for k in range(num_convs):
            layer = Conv2d(input_channels if k == 0 else feature_channels,
                           feature_channels, 3, stride=1, padding=1)
            # Caffe2 implementation uses MSRAFill, which in fact
            # corresponds to kaiming_normal_ in PyTorch
            # nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            # nn.init.constant_(layer.bias, 0)
            self.add_module('mask_fcn{}'.format(k), layer)
            self.blocks.append(layer)

        self.deconv = ConvTranspose2d(feature_channels if num_convs > 0 else input_channels,
                                      feature_channels, 2, 2, 0)
        self.predictor = Conv2d(feature_channels, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            # print("INIT", name)
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
