#pragma once

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif


at::Tensor compute_flow(const at::Tensor& boxes,
                        const int height,
                        const int width) {
  if (boxes.type().is_cuda()) {
    // TODO raise error if not compiled with cuda
    return compute_flow_cuda(boxes, height, width);
  }
  AT_ERROR("Not implemented on the CPU");
}


