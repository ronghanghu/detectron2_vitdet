import torch

def roi_pool2d(input, rois, pooled_height, pooled_width, spatial_scale, return_indices=False):
    ret = torch._C._nn.RoiPooling2d(
        input, rois, pooled_height, pooled_width, spatial_scale)
    return ret if return_indices else ret[0]
