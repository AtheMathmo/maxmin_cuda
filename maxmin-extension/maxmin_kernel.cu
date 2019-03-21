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
    size_t outer_size,
    size_t axis_length,
    size_t inner_stride,
    scalar_t* __restrict__ output) {
  const int axis_index = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  const int inner_index = blockIdx.y * blockDim.y + threadIdx.y;
  const int outer_index = blockIdx.z * blockDim.z + threadIdx.z;
  const int x_stride = 2 * blockDim.x * gridDim.x;
  const int y_stride = blockDim.y * gridDim.y;
  const int z_stride = blockDim.z * gridDim.z;
  
  // 3D grid-stride loop to cover whole array
  for (int z = outer_index; z < outer_size; z += z_stride) {
    for (int x = axis_index; x < axis_length - 1; x += x_stride) {
      for (int y = inner_index; y < inner_stride; y += y_stride) {
        const int pair_index = y + inner_stride *  (x + z * axis_length);
        if (input[pair_index] > input[pair_index + inner_stride]) {
          output[pair_index] = input[pair_index];
          output[pair_index + inner_stride] = input[pair_index + inner_stride];
        } else {
          output[pair_index + inner_stride] = input[pair_index];
          output[pair_index] = input[pair_index + inner_stride];
        }
      }
    }
  }
}


at::Tensor maxmin_cuda_forward(
    at::Tensor input,
    int32_t axis) {
  const int tile_dim = 32;

  const auto num_dims = input.ndimension();
  const auto axis_length = input.size(axis);
  const int true_axis = (axis == -1) ? num_dims - 1 : axis;

  int outer_size = 1;
  for (int i = 0; i < true_axis; ++i) {
    outer_size *= input.size(i);
  }

  const auto grid_dim = std::min((axis_length + tile_dim - 1) / tile_dim, 32LL);
  dim3 grid(grid_dim);
  dim3 block(tile_dim * tile_dim / 2);
  if (num_dims >= 2) {
    block.x = tile_dim / 2;
    //TODO: This might be too large. Need to cap it and compute indices correctly in the kernel.
    block.y = outer_size;
  }

  auto output = at::zeros_like(input);
  if (axis == -1 || axis == (num_dims - 1)) {
    AT_DISPATCH_ALL_TYPES(input.type(), "maxmin_forward_cuda", ([&] {
      maxmin_cuda_forward_kernel_lastdim<scalar_t><<<grid, block>>>(
          input.data<scalar_t>(),
          outer_size,
          axis_length,
          output.data<scalar_t>());
    }));
  } else {
    int inner_stride = 1;
    for (int i = true_axis + 1; i < num_dims; i++) {
      inner_stride *= input.size(i);
    }

    AT_DISPATCH_ALL_TYPES(input.type(), "maxmin_forward_cuda", ([&] {
      maxmin_cuda_forward_kernel<scalar_t><<<grid, block>>>(
          input.data<scalar_t>(),
          outer_size,
          axis_length,
          inner_stride,
          output.data<scalar_t>());
    }));
  }

  return output;
}

template <typename scalar_t>
__global__ void maxmin_cuda_backward_kernel_lastdim(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ grad,
    size_t outer_size,
    size_t axis_length,
    scalar_t* __restrict__ output_grad) {
  const int column_index = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  const int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  const int pair_index = axis_length * row_index + column_index;
  const int stride = 2 * blockDim.x * gridDim.x;
  for (int i = pair_index; i + 1 < (row_index + 1) * axis_length; i += stride)
  {
    if (input[i] > input[i + 1]) {
      output_grad[i] = grad[i];
      output_grad[i + 1] = grad[i + 1];
    } else {
      output_grad[i + 1] = grad[i];
      output_grad[i] = grad[i + 1];
    }
  }
}

template <typename scalar_t>
__global__ void maxmin_cuda_backward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ grad,
    size_t outer_size,
    size_t axis_length,
    size_t inner_stride,
    scalar_t* __restrict__ output_grad) {
  const int axis_index = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  const int inner_index = blockIdx.y * blockDim.y + threadIdx.y;
  const int outer_index = blockIdx.z * blockDim.z + threadIdx.z;
  const int x_stride = 2 * blockDim.x * gridDim.x;
  const int y_stride = blockDim.y * gridDim.y;
  const int z_stride = blockDim.z * gridDim.z;
  
  // 3D grid-stride loop to cover whole array
  for (int z = outer_index; z < outer_size; z += z_stride) {
    for (int x = axis_index; x < axis_length - 1; x += x_stride) {
      for (int y = inner_index; y < inner_stride; y += y_stride) {
        const int pair_index = y + inner_stride *  (x + z * axis_length);
        if (input[pair_index] > input[pair_index + inner_stride]) {
          output_grad[pair_index] = grad[pair_index];
          output_grad[pair_index + inner_stride] = grad[pair_index + inner_stride];
        } else {
          output_grad[pair_index + inner_stride] = grad[pair_index];
          output_grad[pair_index] = grad[pair_index + inner_stride];
        }
      }
    }
  }
}

at::Tensor maxmin_cuda_backward(
    at::Tensor input,
    at::Tensor grad,
    int32_t axis) {
  const int tile_dim = 32;

  const auto num_dims = input.ndimension();
  const auto axis_length = input.size(axis);
  const int true_axis = (axis == -1) ? num_dims - 1 : axis;

  int outer_size = 1;
  for (int i = 0; i < true_axis; ++i) {
    outer_size *= input.size(i);
  }

  const auto grid_dim = std::min((axis_length + tile_dim - 1) / tile_dim, 32LL);
  dim3 grid(grid_dim);
  dim3 block(tile_dim * tile_dim / 2);
  if (num_dims >= 2) {
    block.x = tile_dim / 2;
    //TODO: This might be too large. Need to cap it and compute indices correctly in the kernel.
    block.y = outer_size;
  }

  auto output_grad = at::zeros_like(grad);
  if (axis == -1 || axis == (num_dims - 1)) {
    AT_DISPATCH_ALL_TYPES(input.type(), "maxmin_backward_cuda", ([&] {
      maxmin_cuda_backward_kernel_lastdim<scalar_t><<<grid, block>>>(
          input.data<scalar_t>(),
          grad.data<scalar_t>(),
          outer_size,
          axis_length,
          output_grad.data<scalar_t>());
    }));
  } else {
    int inner_stride = 1;
    for (int i = true_axis + 1; i < num_dims; i++) {
      inner_stride *= input.size(i);
    }

    AT_DISPATCH_ALL_TYPES(input.type(), "maxmin_forward_cuda", ([&] {
      maxmin_cuda_backward_kernel<scalar_t><<<grid, block>>>(
          input.data<scalar_t>(),
          grad.data<scalar_t>(),
          outer_size,
          axis_length,
          inner_stride,
          output_grad.data<scalar_t>());
    }));
  }

  return output_grad;
}
