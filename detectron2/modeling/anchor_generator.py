import copy
import math
import numpy as np
import torch
from torch import nn

from detectron2.structures import Boxes
from detectron2.utils.registry import Registry

ANCHOR_GENERATOR_REGISTRY = Registry("ANCHOR_GENERATOR")


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


@ANCHOR_GENERATOR_REGISTRY.register()
class DefaultAnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set of anchors.
    """

    def __init__(self, cfg):
        super().__init__()
        # fmt: off
        sizes           = cfg.MODEL.RPN.ANCHOR_SIZES
        aspect_ratios   = cfg.MODEL.RPN.ANCHOR_ASPECT_RATIOS
        feature_strides = dict(cfg.MODEL.BACKBONE.COMPUTED_OUT_FEATURE_STRIDES)
        self.strides    = [feature_strides[f] for f in cfg.MODEL.RPN.IN_FEATURES]
        # fmt: on
        self.cell_anchors = self.calculate_anchors(sizes, aspect_ratios, self.strides)

    def calculate_anchors(self, sizes, aspect_ratios, feature_strides):
        """
        Args:
            sizes (list[list[number]]): sizes[i] is the list of anchor sizes to use
                for the i-th feature map. If len(sizes) == 1, then the same list of
                anchor sizes, given by sizes[0], is used for all feature maps. Anchor
                sizes are given in absolute lengths in units of the input image;
                they do not dynamically scale if the input image size changes.
            aspect_ratios (list[list[number]]): aspect_ratios[i] is the list of
                anchor aspect ratios to use for the i-th feature map. If
                len(aspect_ratios) == 1, then the same list of anchor aspect ratios,
                given by aspect_ratios[0], is used for all feature maps.
            feature_strides (list[number]): list of feature map strides (with respect
                to the input image) for each input feature map.
        """

        # If one size (or aspect ratio) is specified and there are multiple feature
        # maps, then we "broadcast" anchors of that single size (or aspect ratio)
        # over all feature maps.
        num_feature_maps = len(feature_strides)
        if len(sizes) == 1:
            sizes *= num_feature_maps
        if len(aspect_ratios) == 1:
            aspect_ratios *= num_feature_maps
        assert num_feature_maps == len(sizes)
        assert num_feature_maps == len(aspect_ratios)

        cell_anchors = [
            self.generate_cell_anchors(s, a, stride).float()
            for s, a, stride in zip(sizes, aspect_ratios, feature_strides)
        ]

        return BufferList(cell_anchors)

    @property
    def num_cell_anchors(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios and 5 sizes, the number of anchors is 15.
                (See also RPN.ANCHOR_SIZES and RPN.ANCHOR_ASPECT_RATIOS in config)

                In standard RPN models, `num_cell_anchors` on every feature map is the same.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):
            grid_height, grid_width = size
            device = base_anchors.device
            shifts_x = torch.arange(
                0, grid_width * stride, step=stride, dtype=torch.float32, device=device
            )
            shifts_y = torch.arange(
                0, grid_height * stride, step=stride, dtype=torch.float32, device=device
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

        return anchors

    def generate_cell_anchors(
        self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2), _unused_stride=None
    ):
        """
        Generate a tensor storing anchor boxes, which are continuous geometric rectangles
        centered on one feature map point sample. We can later build the set of anchors
        for the entire feature map by tiling these tensors; see `meth:grid_anchors`.

        Args:
            sizes (tuple[float]): Absolute size of the anchors in the units of the input
                image (the input received by the network, after ungoing necessary scaling).
                The absolute size is given as the side length of a box.
            aspect_ratios (tuple[float]]): Aspect ratios of the boxes computed as box
                height / width.

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
                in XYXY format.
        """
        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                # s * s = w * h
                # a = h / w
                # ... some algebra ...
                # w = sqrt(s * s / a)
                # h = a * w
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return torch.tensor(anchors)

    def forward(self, image_list, feature_maps):
        """
        Returns:
            list[list[Boxes]]: a list of #image elements. Each is a list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
        """
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)

        anchors_in_image = []
        for anchors_per_feature_map in anchors_over_all_feature_maps:
            boxes = Boxes(anchors_per_feature_map)
            anchors_in_image.append(boxes)

        anchors = [copy.deepcopy(anchors_in_image) for _ in range(len(image_list))]
        return anchors


@ANCHOR_GENERATOR_REGISTRY.register()
class OriginalRPNAnchorGenerator(DefaultAnchorGenerator):
    """
    The anchor generator that was defined in the original Faster R-CNN code.
    Models using this generator yield the same AP as those using the new default one,
    however this version defines cell anchors in a less natural way with a shift
    relative to the feature grid and quantization that results in slightly different
    sizes for the different aspect ratios.
    """

    def generate_cell_anchors(
        self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2), stride=16
    ):
        return _original_rpn_generate_anchors(sizes, aspect_ratios, stride)


@ANCHOR_GENERATOR_REGISTRY.register()
class RetinaNetAnchorGenerator(DefaultAnchorGenerator):
    """
    For a set of image sizes and feature maps, computes a set of anchors for RetinaNet.
    """

    def __init__(self, cfg):
        nn.Module.__init__(self)
        # fmt: off
        sizes           = cfg.MODEL.RETINANET.ANCHOR_SIZES
        aspect_ratios   = cfg.MODEL.RETINANET.ANCHOR_ASPECT_RATIOS
        feature_strides = dict(cfg.MODEL.BACKBONE.COMPUTED_OUT_FEATURE_STRIDES)
        self.strides    = [feature_strides[f] for f in cfg.MODEL.RETINANET.IN_FEATURES]
        # fmt: on
        self.cell_anchors = self.calculate_anchors(sizes, aspect_ratios, self.strides)


def build_anchor_generator(cfg):
    anchor_generator = cfg.MODEL.ANCHOR_GENERATOR.NAME
    return ANCHOR_GENERATOR_REGISTRY.get(anchor_generator)(cfg)


#
# Code for the original RPN anchor generation method is below. We no longer use
# this by default.
#

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------


# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#        [-175.,  -87.,  192.,  104.],
#        [-359., -183.,  376.,  200.],
#        [ -55.,  -55.,   72.,   72.],
#        [-119., -119.,  136.,  136.],
#        [-247., -247.,  264.,  264.],
#        [ -35.,  -79.,   52.,   96.],
#        [ -79., -167.,   96.,  184.],
#        [-167., -343.,  184.,  360.]])


def _original_rpn_generate_anchors(
    sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2), stride=16
):
    """
    Generates anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.

    Returns:
        Tensor of shape Nx4, where `N == len(sizes) * len(aspect_ratios)`.
    """
    return _generate_anchors(
        stride, np.array(sizes, dtype=np.float) / stride, np.array(aspect_ratios, dtype=np.float)
    )


def _generate_anchors(base_size, scales, aspect_ratios):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack([_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])])
    return torch.from_numpy(anchors)


def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        )
    )
    return anchors


def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors
