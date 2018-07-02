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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_flow", &compute_flow, "compute_flow");
}
 
