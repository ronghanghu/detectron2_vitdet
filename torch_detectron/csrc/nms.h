#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif


at::Tensor nms(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold) {

  if (dets.type().is_cuda()) {
    // TODO raise error if not compiled with CUDA
    if (dets.numel() == 0)
      return torch::CPU(at::kLong).tensor();
    auto b = at::cat({dets, scores.unsqueeze(1)}, 1);
    return nms_cuda(b, threshold);
  }

  at::Tensor result = nms_cpu(dets, scores, threshold);
  return result;
}
