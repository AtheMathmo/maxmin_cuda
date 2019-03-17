#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

at::Tensor maxmin_cuda_forward(
    at::Tensor input,
    int32_t axis);

at::Tensor maxmin_cuda_backward(
    at::Tensor input,
    at::Tensor grad,
    int32_t axis);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor maxmin_forward(
    at::Tensor input,
    int32_t axis) {
  CHECK_INPUT(input);
  return maxmin_cuda_forward(input, axis);
}

at::Tensor maxmin_backward(
    at::Tensor input,
    at::Tensor grad,
    int32_t axis) {
  CHECK_INPUT(input);
  CHECK_INPUT(grad);

  return maxmin_cuda_backward(
      input,
      grad,
      axis);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &maxmin_forward, "MaxMin forward (CUDA)");
  m.def("backward", &maxmin_backward, "MaxMin backward (CUDA)");
}