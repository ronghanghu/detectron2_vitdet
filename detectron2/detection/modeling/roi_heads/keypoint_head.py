import torch
from torch import nn
from torch.nn import functional as F

from detectron2 import layers
from detectron2.layers import Conv2d, cat
from detectron2.structures import Keypoints, heatmaps_to_keypoints
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

_TOTAL_SKIPPED = 0

ROI_KEYPOINT_HEAD_REGISTRY = Registry("ROI_KEYPOINT_HEAD")


def build_keypoint_head(cfg):
    name = cfg.MODEL.ROI_KEYPOINT_HEAD.NAME
    return ROI_KEYPOINT_HEAD_REGISTRY.get(name)(cfg)


def keypoint_rcnn_loss(pred_keypoint_logits, instances, normalizer):
    """
    Arguments:
        pred_keypoint_logits (Tensor): A tensor of shape (B, N, S, S) where B is the batch size,
            N is the number of keypoints, and S is the side length of the keypoint heatmap. The
            values are spatial logits.
        instances (list[Instances]): A list of N Instances, where N is the batch size.
            These instances are predictions from the model
            that are in 1:1 correspondence with pred_keypoint_logits.
            Each Instances should contain a `gt_keypoints` field containing a `structures.Keypoint`
            instance.
        normalizer (float): Normalize the loss by this amount.
            If not specified, we normalize by the number of visible keypoints in the minibatch.

    Returns a scalar tensor containing the loss.
    """
    heatmaps = []
    valid = []

    keypoint_side_len = pred_keypoint_logits.shape[2]
    for instances_per_image in instances:
        keypoints = instances_per_image.gt_keypoints
        heatmaps_per_image, valid_per_image = keypoints.to_heatmap(
            instances_per_image.proposal_boxes.tensor, keypoint_side_len
        )
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))

    keypoint_targets = cat(heatmaps, dim=0)
    valid = cat(valid, dim=0).to(dtype=torch.uint8)
    valid = torch.nonzero(valid).squeeze(1)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if keypoint_targets.numel() == 0 or valid.numel() == 0:
        global _TOTAL_SKIPPED
        _TOTAL_SKIPPED += 1
        storage = get_event_storage()
        storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
        return pred_keypoint_logits.sum() * 0

    N, K, H, W = pred_keypoint_logits.shape
    pred_keypoint_logits = pred_keypoint_logits.view(N * K, H * W)

    keypoint_loss = F.cross_entropy(
        pred_keypoint_logits[valid], keypoint_targets[valid], reduction="sum"
    )

    # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
    if normalizer is None:
        normalizer = valid.numel()
    keypoint_loss /= normalizer

    return keypoint_loss


def keypoint_rcnn_inference(pred_keypoint_logits, pred_instances):
    """
    Post process each predicted keypoint heatmap in `pred_keypoint_logits` into (x, y, score, prob)
        and add it to the `pred_instances` as a `pred_keypoints` field.

    Args:
        pred_keypoint_logits (Tensor): A tensor of shape (B, N, S, S) where B is the batch size,
            N is the number of keypoints, and S is the side length of the keypoint heatmap. The
            values are spatial logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the batch size.

    Returns:
        None. boxes will contain an extra "pred_keypoints" field.
    """
    # flatten all bboxes from all images together (list[Boxes] -> Nx4 tensor)
    bboxes_flat = cat([b.pred_boxes.tensor for b in pred_instances], dim=0)

    keypoint_results = heatmaps_to_keypoints(
        pred_keypoint_logits.cpu().detach().numpy(), bboxes_flat.cpu().detach().numpy()
    )
    keypoint_results = torch.from_numpy(keypoint_results).to(pred_keypoint_logits.device)
    num_instances_per_image = [len(i) for i in pred_instances]
    keypoint_results = keypoint_results.split(num_instances_per_image, dim=0)

    for keypoint_results_per_image, instances_per_image in zip(keypoint_results, pred_instances):
        # keypoint_results_per_image is (num instances)x(num keypoints)x(x, y, score, prob)
        vis = torch.ones(
            len(instances_per_image),
            keypoint_results_per_image.size(1),
            1,
            device=keypoint_results_per_image.device,
        )
        keypoint_xy = keypoint_results_per_image[:, :, :2]
        keypoint_xyv = cat((keypoint_xy, vis), dim=2)
        instances_per_image.pred_keypoints = Keypoints(keypoint_xyv)


@ROI_KEYPOINT_HEAD_REGISTRY.register()
class KRCNNConvDeconvUpsampleHead(nn.Module):
    """
    A standard keypoint head containing a series of 3x3 convs, followed by
    a transpose convolution and bilinear interpolation for upsampling.
    """

    def __init__(self, cfg):
        """
        The following attributes are parsed from config:
            in_channels: from ROI_KEYPOINT_HEAD.COMPUTED_INPUT_SIZE
            conv_dims: an iterable of output channel counts for each conv in the head
                         e.g. (512, 512, 512) for three convs outputting 512 channels.
            num_keypoints: number of keypoint heatmaps to predicts, determines the number of
                           channels in the final output.
        """
        super(KRCNNConvDeconvUpsampleHead, self).__init__()

        # fmt: off
        # default up_scale to 2 (this can eventually be moved to config)
        up_scale      = 2
        conv_dims     = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS
        num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
        in_channels   = cfg.MODEL.ROI_KEYPOINT_HEAD.COMPUTED_INPUT_SIZE[0]
        # fmt: on

        self.blocks = []
        for idx, layer_channels in enumerate(conv_dims, 1):
            module = Conv2d(in_channels, layer_channels, 3, stride=1, padding=1)
            self.add_module("conv_fcn{}".format(idx), module)
            self.blocks.append(module)
            in_channels = layer_channels

        deconv_kernel = 4
        self.score_lowres = layers.ConvTranspose2d(
            in_channels, num_keypoints, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
        )
        self.up_scale = up_scale

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
        x = self.score_lowres(x)
        x = layers.interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
        return x