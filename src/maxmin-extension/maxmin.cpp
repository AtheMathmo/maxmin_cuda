#include <torch/torch.h>

#include <iostream>

at::Tensor sigmoid(at::Tensor z) {
  auto s = at::sigmoid(z);
  return s;
}

at::Tensor d_sigmoid(at::Tensor z) {
  auto s = at::sigmoid(z);
  return (1 - s) * s;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sigmoid, "Sigmoid forward");
  m.def("backward", &d_sigmoid, "Sigmoid backward");
}