import math
import torch

from lib.utils.generate_anchors import generate_anchors


def meshgrid(x, y=None):
    if y is None:
        y = x
    m, n = x.size(0), y.size(0)
    grid_x = x[None].expand(n, m).contiguous()
    grid_y = y[:, None].expand(n, m).contiguous()
    return grid_x, grid_y


class AnchorGridGenerator(object):
    def __init__(self, stride, sizes, aspect_ratios, max_size):
        cell_anchors = generate_anchors(stride, sizes, aspect_ratios).float()
        self.stride = stride
        self.cell_anchors = cell_anchors
        self.max_size = max_size

    def generate(self, coarsest_stride=None):
        coarsest_stride = self.stride  # TODO fix
        fpn_max_size = coarsest_stride * math.ceil(
            self.max_size / float(coarsest_stride)
        )
        # TODO not good because stride is already in cell_anchors
        field_size = int(math.ceil(fpn_max_size / float(self.stride)))
        shifts = torch.arange(0, field_size) * self.stride
        shift_x, shift_y = meshgrid(shifts, shifts)
        shift_x = shift_x.view(-1)
        shift_y = shift_y.view(-1)
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

        A = self.cell_anchors.size(0)
        K = shifts.size(0)
        field_of_anchors = (
            self.cell_anchors.view(1, A, 4) +
            shifts.view(1, K, 4).permute(1, 0, 2).to(dtype=torch.float32)  # TODO remove casting
        )
        # field_of_anchors = field_of_anchors.view(K * A, 4)
        field_of_anchors = field_of_anchors.view(field_size, field_size, A, 4)

        return field_of_anchors



