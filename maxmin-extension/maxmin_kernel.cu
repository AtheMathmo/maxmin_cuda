#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void maxmin_cuda_forward_kernel(
    const scalar_t* __restrict__ input,
    size_t outer_size,
    size_t axis_length,
    size_t inner_stride,
    scalar_t* __restrict__ output) {
  const int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int axis_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const int stride_idx = blockIdx.z * blockDim.z + threadIdx.z;
  const int inner_idx = 2 * axis_idx;
  if (outer_idx < outer_size && stride_idx < inner_stride) {
    const int left_idx = inner_stride * axis_length * outer_idx + inner_idx * inner_stride + stride_idx;
    if (inner_idx < axis_length - 1) {
      const int right_idx = left_idx + inner_stride;
      if (input[left_idx] > input[right_idx]) {
        output[left_idx] = input[left_idx];
        output[right_idx] = input[right_idx];
      } else {
        output[left_idx] = input[right_idx];
        output[right_idx] = input[left_idx];
      }
    } else if (inner_idx < axis_length) {
      // In range, but at end of sorting axis
      output[left_idx] = input[left_idx];
    }
  }
}


at::Tensor maxmin_cuda_forward(
    at::Tensor input,
    int32_t axis) {
  const auto num_dims = input.ndimension();
  const auto axis_length = input.size(axis);
  const int true_axis = (axis == -1) ? num_dims - 1 : axis;

  int outer_size = 1;
  for (int i = 0; i < true_axis; ++i) {
    outer_size *= input.size(i);
  };
  int inner_stride = 1;
  for (int i = true_axis + 1; i < num_dims; i++) {
    inner_stride *= input.size(i);
  }

  dim3 block(8, 8, 8);
  dim3 grid((outer_size + 7) / 8, (axis_length + 15) / 16, (inner_stride + 7) / 8);

  auto output = at::zeros_like(input);
  AT_DISPATCH_ALL_TYPES(input.type(), "maxmin_forward_cuda", ([&] {
    maxmin_cuda_forward_kernel<scalar_t><<<grid, block>>>(
        input.data<scalar_t>(),
        outer_size,
        axis_length,
        inner_stride,
        output.data<scalar_t>());
  }));
  return output;
}



template <typename scalar_t>
__global__ void maxmin_cuda_backward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ grad,
    size_t outer_size,
    size_t axis_length,
    size_t inner_stride,
    scalar_t* __restrict__ output_grad) {
  const int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int axis_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const int stride_idx = blockIdx.z * blockDim.z + threadIdx.z;
  const int inner_idx = 2 * axis_idx;
  if (outer_idx < outer_size && stride_idx < inner_stride) {
    const int left_idx = inner_stride * axis_length * outer_idx + inner_idx * inner_stride + stride_idx;
    if (inner_idx < axis_length - 1) {
      const int right_idx = left_idx + inner_stride;
      if (input[left_idx] > input[right_idx]) {
        output_grad[left_idx] = grad[left_idx];
        output_grad[right_idx] = grad[right_idx];
      } else {
        output_grad[left_idx] = grad[right_idx];
        output_grad[right_idx] = grad[left_idx];
      }
    } else if (inner_idx < axis_length) {
      // In range, but at end of sorting axis
      output_grad[left_idx] = grad[left_idx];
    }
  }
}

at::Tensor maxmin_cuda_backward(
    at::Tensor input,
    at::Tensor grad,
    int32_t axis) {
  const auto num_dims = input.ndimension();
  const auto axis_length = input.size(axis);
  const int true_axis = (axis == -1) ? num_dims - 1 : axis;

  int outer_size = 1;
  for (int i = 0; i < true_axis; ++i) {
    outer_size *= input.size(i);
  };
  int inner_stride = 1;
  for (int i = true_axis + 1; i < num_dims; i++) {
    inner_stride *= input.size(i);
  }

  dim3 block(8, 8, 8);
  dim3 grid((outer_size + 7) / 8, (axis_length + 15) / 16, (inner_stride + 7) / 8);

  auto output_grad = at::zeros_like(grad);

  AT_DISPATCH_ALL_TYPES(input.type(), "maxmin_forward_cuda", ([&] {
    maxmin_cuda_backward_kernel<scalar_t><<<grid, block>>>(
        input.data<scalar_t>(),
        grad.data<scalar_t>(),
        outer_size,
        axis_length,
        inner_stride,
        output_grad.data<scalar_t>());
  }));
  return output_grad;
}

