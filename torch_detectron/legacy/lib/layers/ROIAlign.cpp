#include <torch/torch.h>

at::Tensor ROIAlign_forward_cuda(const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int sampling_ratio);

at::Tensor ROIAlign_backward_cuda(const at::Tensor& grad,
                                  const at::Tensor& input,
                                  const at::Tensor& rois,
                                  const float spatial_scale,
                                  const int pooled_height,
                                  const int pooled_width,
                                  const int sampling_ratio);

// ROIPool
std::tuple<at::Tensor, at::Tensor> ROIPool_forward_cuda(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width);

at::Tensor ROIPool_backward_cuda(const at::Tensor& grad,
                                 const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const at::Tensor& argmax,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width);


// Interface for Python
at::Tensor ROIAlign_forward(const at::Tensor& input,
                            const at::Tensor& rois,
                            const float spatial_scale,
                            const int pooled_height,
                            const int pooled_width,
                            const int sampling_ratio) {
  if (input.type().is_cuda())
    return ROIAlign_forward_cuda(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
  std::runtime_error("Not implemented on the CPU");
}

at::Tensor ROIAlign_backward(const at::Tensor& grad,
                             const at::Tensor& input,
                             const at::Tensor& rois,
                             const float spatial_scale,
                             const int pooled_height,
                             const int pooled_width,
                             const int sampling_ratio) {
  if (grad.type().is_cuda())
    return ROIAlign_backward_cuda(grad, input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
  std::runtime_error("Not implemented on the CPU");

}


std::tuple<at::Tensor, at::Tensor> ROIPool_forward(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width) {
  if (input.type().is_cuda())
    return ROIPool_forward_cuda(input, rois, spatial_scale, pooled_height, pooled_width);
  std::runtime_error("Not implemented on the CPU");
}

at::Tensor ROIPool_backward(const at::Tensor& grad,
                                 const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const at::Tensor& argmax,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width) {
  if (grad.type().is_cuda())
    return ROIPool_backward_cuda(grad, input, rois, argmax, spatial_scale, pooled_height, pooled_width);
  std::runtime_error("Not implemented on the CPU");

}




PYBIND11_MODULE(detectron_modules, m) {
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
  m.def("roi_pool_forward", &ROIPool_forward, "ROIPool_forward");
  m.def("roi_pool_backward", &ROIPool_backward, "ROIPool_backward");
}
