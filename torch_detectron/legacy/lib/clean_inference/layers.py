import torch
from torch import nn

from torch.autograd import Function
from torch.autograd.function import once_differentiable

from torch.nn.modules.utils import _pair


def _load_C_extensions():
    import os.path
    from torch.utils.cpp_extension import load as load_ext

    this_dir = os.path.dirname(os.path.abspath(__file__))
    source = [
        'ROIAlign.cpp',
        'ROIAlign_cpu.cpp',
    ]
    extra_cflags = []
    if torch.cuda.is_available():
        source.append('ROIAlign_cuda.cu')
        extra_cflags = ['-DWITH_CUDA']
    source = [os.path.join(this_dir, s) for s in source]
    return load_ext('roi_align', source, extra_cflags=extra_cflags)

_C = _load_C_extensions()


class _ROIAlign(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(input, roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        output = _C.roi_align_forward(input, roi, spatial_scale,
                output_size[0], output_size[1], sampling_ratio)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, rois = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        grad_input =  _C.roi_align_backward(grad_output, input, rois, spatial_scale,
                output_size[0], output_size[1], sampling_ratio)
        return grad_input, None, None, None, None

roi_align = _ROIAlign.apply

class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input, rois):
        return roi_align(input, rois, self.output_size, self.spatial_scale, self.sampling_ratio)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'output_size=' + str(self.output_size)
        tmpstr += ', spatial_scale=' + str(self.spatial_scale)
        tmpstr += ', sampling_ratio=' + str(self.sampling_ratio)
        tmpstr += ')'
        return tmpstr


class FixedBatchNorm2d(nn.Module):
    def __init__(self, n):
        super(FixedBatchNorm2d, self).__init__()
        self.register_buffer('scale', torch.zeros(1, n, 1, 1))
        self.register_buffer('bias', torch.zeros(1, n, 1, 1))

    def forward(self, x):
        return x * self.scale + self.bias

    def __repr__(self):
        return self.__class__.__name__ + '(n=' + str(self.scale.size(1)) + ')'
