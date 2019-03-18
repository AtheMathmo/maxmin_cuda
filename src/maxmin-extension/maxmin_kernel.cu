#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void maxmin_cuda_forward_kernel_lastdim(
    const scalar_t* __restrict__ input,
    size_t outer_size,
    size_t axis_length,
    scalar_t* __restrict__ output) {
  // TODO: Compute the pair indices correctly, over arbitrary axis.
  // Probably want to design grid accordingly for cache optimization
  const int column_index = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  const int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  const int pair_index = axis_length * row_index + column_index;
  const int stride = 2 * blockDim.x * gridDim.x;
  for (int i = pair_index; i + 1 < (row_index + 1) * axis_length; i += stride)
  {
    if (input[i] > input[i + 1]) {
      output[i] = input[i];
      output[i + 1] = input[i + 1];
    } else {
      output[i + 1] = input[i];
      output[i] = input[i + 1];
    }
  }
}

template <typename scalar_t>
__global__ void maxmin_cuda_forward_kernel(
    const scalar_t* __restrict__ input,
    const int axis,
    size_t axis_length,
    scalar_t* __restrict__ output) {
  throw "Not implemented";
}

at::Tensor maxmin_cuda_forward(
    at::Tensor input,
    int32_t axis) {
  const int tile_dim = 32;
  // TODO: Make sure this is safe.
  const auto num_dims = input.ndimension();
  const auto axis_length = input.size(axis);

  if (num_dims > 2) {
    throw "MaxMin CUDA only works for 2 or fewer dims";
  }

  int outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= input.size(i);
  }

  dim3 grid(tile_dim);
  dim3 block(tile_dim / 2);
  if (num_dims == 2) {
    block.y = input.size(0);
  }

  auto output = at::zeros_like(input);
  if (axis == -1 || axis == (num_dims - 1)) {
    AT_DISPATCH_FLOATING_TYPES(input.type(), "maxmin_forward_cuda", ([&] {
      maxmin_cuda_forward_kernel_lastdim<scalar_t><<<grid, block>>>(
          input.data<scalar_t>(),
          outer_size,
          axis_length,
          output.data<scalar_t>());
    }));
  } else {
    throw "Currently only support acting on last dim";
  }

  return output;
}

template <typename scalar_t>
__global__ void maxmin_cuda_backward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ grad,
    const int axis,
    scalar_t* __restrict__ output_grad,
    size_t axis_length) {
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

  const int threads = 1024;
  // TODO: Make sure this is safe.
  const auto axis_length = input.size(axis);
  const int blocks = (axis_length + threads - 1) / threads;

  auto output_grad = at::zeros_like(grad);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "maxmin_backward_cuda", ([&] {
    maxmin_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        input.data<scalar_t>(),
        grad.data<scalar_t>(),
        axis,
        output_grad.data<scalar_t>(),
        axis_length);
  }));
  return output_grad;
}
