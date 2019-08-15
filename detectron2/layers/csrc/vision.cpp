#include "ROIAlign/ROIAlign.h"
#include "deformable/deform_conv.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");

  m.def("deform_conv_forward",
        &deform_conv_forward, "deform_conv_forward");
  m.def("deform_conv_backward_input",
        &deform_conv_backward_input, "deform_conv_backward_input");
  m.def("deform_conv_backward_filter",
        &deform_conv_backward_filter, "deform_conv_backward_filter");
  m.def("modulated_deform_conv_forward",
        &modulated_deform_conv_forward, "modulated_deform_conv_forward");
  m.def("modulated_deform_conv_backward",
        &modulated_deform_conv_backward, "modulated_deform_conv_backward");
}
