#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void maxmin_cuda_forward_kernel(
    const scalar_t* __restrict__ input,
    const int axis,
    scalar_t* __restrict__ output,
    size_t axis_length) {
  // TODO: Compute the pair indices correctly, over arbitrary axis.
  // Probably want to design grid accordingly for cache optimization
  const int pair_index = blockIdx.x * blockDim.x + 2*threadIdx.x;
  if (pair_index + 1 < axis_length) {
    if (input[pair_index] > input[pair_index + 1]) {
      output[pair_index] = input[pair_index];
      output[pair_index + 1] = input[pair_index + 1];
    } else {
      output[pair_index + 1] = input[pair_index];
      output[pair_index] = input[pair_index + 1];
    }
  }
}

at::Tensor maxmin_cuda_forward(
    at::Tensor input,
    int32_t axis) {
  const int threads = 1024;
  // TODO: Make sure this is safe.
  const auto axis_length = input.size(axis);
  const int blocks = (axis_length + threads - 1) / threads;

  auto output = at::zeros_like(input);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "maxmin_forward_cuda", ([&] {
    maxmin_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        input.data<scalar_t>(),
        axis,
        output.data<scalar_t>(),
        axis_length);
  }));
  return output;
}

template <typename scalar_t>
__global__ void maxmin_cuda_backward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ grad,
    const int axis,
    scalar_t* __restrict__ output_grad) {
  const int pair_index = blockIdx.x * blockDim.x + 2*threadIdx.x;
  if (pair_index + 1 < axis_length) {
    if (input[pair_index] > input[pair_index + 1]) {
      output_grad[pair_index] = grad[pair_index];
      output_grad[pair_index + 1] = grad[pair_index + 1];
    } else {
      output_grad[pair_index + 1] = grad[pair_index];
      output_grad[pair_index] = grad[pair_index + 1];
    }
  }
}

at::Tensor maxmin_cuda_backward(
    at::Tensor input,
    at::Tensor grad,
    int32_t axis) {

  return grad;
}



