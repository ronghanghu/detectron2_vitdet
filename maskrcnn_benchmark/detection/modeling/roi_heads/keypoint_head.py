import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark import layers
from maskrcnn_benchmark.layers import Conv2d, cat
from maskrcnn_benchmark.structures.keypoints import Keypoints, heatmaps_to_keypoints
from maskrcnn_benchmark.utils.events import get_event_storage


_TOTAL_SKIPPED = 0


def keypoint_rcnn_loss(keypoint_logits_pred, box_lists_pred, keypoint_side_len):
    """
    Arguments:
        keypoint_logits_pred (Tensor): A tensor of shape (B, N, S, S) where B is the batch size,
            N is the number of keypoints, and S is the side length of the keypoint heatmap. The
            values are spatial logits.
        proposals (list[BoxList]): A list of N BoxLists, where N is the batch size. These boxes
            are predictions from the model that are in 1:1 correspondence with keypoint_logits_pred.
            Each BoxList should contain a `gt_keypoints` field containing a `structures.Keypoint`
            instance.
        keypoint_side_len (int): Specifies the side length of the square keypoint heatmaps.

    Returns a scalar tensor containing the loss.
    """
    heatmaps = []
    valid = []

    for proposals_per_image in box_lists_pred:
        keypoints = proposals_per_image.get_field("gt_keypoints")
        heatmaps_per_image, valid_per_image = keypoints.to_heatmap(
            proposals_per_image, keypoint_side_len
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
        return keypoint_logits_pred.sum() * 0

    N, K, H, W = keypoint_logits_pred.shape
    keypoint_logits_pred = keypoint_logits_pred.view(N * K, H * W)

    keypoint_loss = F.cross_entropy(
        keypoint_logits_pred[valid], keypoint_targets[valid], reduction="mean"
    )
    return keypoint_loss


def keypoint_rcnn_inference(keypoint_logits_pred, box_lists_pred):
    """
    Post process each predicted keypoint heatmap in `keypoint_logits_pred` into (x, y, score, prob)
        and add it to the boxlist as a `pred_keypoints` field.

    Args:
        keypoint_logits_pred (Tensor): A tensor of shape (B, N, S, S) where B is the batch size,
            N is the number of keypoints, and S is the side length of the keypoint heatmap. The
            values are spatial logits.
        box_lists_pred (list[BoxList]): A list of N BoxLists, where N is the batch size.

    Returns:
        None. boxes will contain an extra "keypoints" field.
    """
    # flatten all bboxes from all images together (list[BoxList] -> Nx4 tensor)
    bboxes_flat = cat([b.bbox for b in box_lists_pred], dim=0)

    keypoint_results = heatmaps_to_keypoints(
        keypoint_logits_pred.detach().cpu().numpy(), bboxes_flat.cpu().numpy()
    )
    keypoint_results = torch.from_numpy(keypoint_results).to(keypoint_logits_pred.device)
    boxes_per_image = [len(box) for box in box_lists_pred]
    keypoint_results = keypoint_results.split(boxes_per_image, dim=0)

    for keypoint_result, box in zip(keypoint_results, box_lists_pred):
        # keypoint_result is (num boxes)x(num keypoints)x(x, y, prob, score)
        vis = torch.ones(len(box), keypoint_result.size(1), 1, device=keypoint_result.device)
        keypoint_xy = keypoint_result[:, :, :2]
        keypoint_xyv = cat((keypoint_xy, vis), dim=2)
        box.add_field("pred_keypoints", Keypoints(keypoint_xyv, box.size))


class KRCNNConvDeconvUpsampleHead(nn.Module):
    """
    A standard keypoint head containing a series of 3x3 convs, followed by
    a transpose convolution and bilinear interpolation for upsampling.
    """

    def __init__(self, in_channels, conv_layers, num_keypoints, up_scale=2):
        """
        Arguments:
            in_channels: number of input channels to the first conv in this head
            conv_layers: an iterable of output channel counts for each conv in the head
                         e.g. (512, 512, 512) for three convs outputting 512 channels.
            num_keypoints: number of keypoint heatmaps to predicts, determines the number of
                           channels in the final output.
            up_scale: scale factor for final bilinear interpolation
                      the spatial size of the final output will be 2*up_scale*input_size
        """
        super(KRCNNConvDeconvUpsampleHead, self).__init__()
        self.blocks = []
        for idx, layer_channels in enumerate(conv_layers, 1):
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


def build_keypoint_head(cfg, input_features):
    conv_layers = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_LAYERS
    num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
    return KRCNNConvDeconvUpsampleHead(input_features, conv_layers, num_keypoints)
