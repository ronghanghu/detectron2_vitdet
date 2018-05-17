#include <torch/torch.h>

at::Tensor compute_flow_cuda(const at::Tensor& boxes,
                             const int height,
                             const int width);

at::Tensor compute_flow(const at::Tensor& boxes,
                        const int height,
                        const int width) {
  if (boxes.type().is_cuda()) {
    return compute_flow_cuda(boxes, height, width);
  }
  AT_ERROR("Not implemented on the CPU");
}

//////////////////

at::Tensor nms_cuda(const at::Tensor boxes, float nms_overlap_thresh);

at::Tensor nms(const at::Tensor boxes, const at::Tensor scores, float nms_overlap_thresh) {
  if (boxes.type().is_cuda()) {
    if (boxes.numel() == 0)
      return torch::CPU(at::kLong).tensor();
    auto b = at::cat({boxes, scores.unsqueeze(1)}, 1);
    return nms_cuda(b, nms_overlap_thresh);
  }
  AT_ERROR("Not implemented on the CPU");
}
 
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_flow", &compute_flow, "compute_flow");
  m.def("nms", &nms, "nms");
}
 
