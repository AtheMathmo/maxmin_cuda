#include <torch/torch.h>

#include <vector>

// CPU functions
template <typename scalar_t>
void maxmin_swap(
    const scalar_t* input,
    const scalar_t* comparison,
    scalar_t* output,
    const int32_t index1,
    const int32_t index2) {
  if (comparison[index1] > comparison[index2]) {
    output[index1] = input[index1];
    output[index2] = input[index2];
  } else {
    output[index2] = input[index1];
    output[index1] = input[index2];
  }
}


//TODO: This can be largely reused in the CUDA code too.
std::vector<int> maxmin_cpu_process(
    at::Tensor input,
    int32_t axis) {
  const auto num_dims = input.ndimension();
  const int axis_length = input.size(axis);
  const int true_axis = (axis == -1) ? num_dims - 1 : axis;

  int outer_size = 1;
  for (int i = 0; i < true_axis; ++i) {
    outer_size *= input.size(i);
  }

  int inner_size = 1;
  for (int i = true_axis + 1; i < num_dims; i++) {
    inner_size *= input.size(i);
  }

  return {outer_size, axis_length, inner_size};
}

// TODO: These implementations are, as expected, slow.
at::Tensor maxmin_cpu_forward(
    at::Tensor input,
    int32_t axis) {
  auto dim_info = maxmin_cpu_process(input, axis);
  auto outer_size = dim_info[0];
  auto axis_length = dim_info[1];
  auto inner_size = dim_info[2];

  auto output = at::zeros_like(input);
  
  for (int x = 0; x < outer_size; x++) {
    for (int y = 0; y < axis_length; y += 2) {
      for (int z = 0; z < inner_size; z++) {
        int pair_index_1 = z + inner_size *  (y + x * axis_length);
        int pair_index_2 = pair_index_1 + inner_size;
        AT_DISPATCH_ALL_TYPES(input.type(),  "maxmin_forward_cpu", ([&] {
          maxmin_swap<scalar_t>(
              input.data<scalar_t>(),
              input.data<scalar_t>(),
              output.data<scalar_t>(),
              pair_index_1,
              pair_index_2);
        }));
      }
    }
  }
  return output;
}

at::Tensor maxmin_cpu_backward(
    at::Tensor input,
    at::Tensor grad,
    int32_t axis) {
  auto dim_info = maxmin_cpu_process(input, axis);
  auto outer_size = dim_info[0];
  auto axis_length = dim_info[1];
  auto inner_size = dim_info[2];

  auto output_grad = at::zeros_like(input);

  for (int x = 0; x < outer_size; x++) {
    for (int y = 0; y < axis_length; y += 2) {
      for (int z = 0; z < inner_size; z++) {
        int pair_index_1 = z + inner_size *  (y + x * axis_length);
        int pair_index_2 = pair_index_1 + inner_size;
        AT_DISPATCH_ALL_TYPES(input.type(),  "maxmin_backward_cpu", ([&] {
          maxmin_swap<scalar_t>(
              grad.data<scalar_t>(),
              input.data<scalar_t>(),
              output_grad.data<scalar_t>(),
              pair_index_1,
              pair_index_2);
        }));
      }
    }
  }
  return output_grad;
}

// CUDA declarations
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
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

at::Tensor maxmin_forward(
    at::Tensor input,
    int32_t axis) {
  CHECK_INPUT(input);
  if (input.type().is_cuda()) {
    return maxmin_cuda_forward(input, axis);
  } else {
    return maxmin_cpu_forward(input, axis);
  }
}

at::Tensor maxmin_backward(
    at::Tensor input,
    at::Tensor grad,
    int32_t axis) {
  CHECK_INPUT(input);
  CHECK_INPUT(grad);

  if (input.type().is_cuda()) {
    return maxmin_cuda_backward(input, grad, axis);
  } else {
    return maxmin_cpu_backward(input, grad, axis);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &maxmin_forward, "MaxMin forward (CUDA)");
  m.def("backward", &maxmin_backward, "MaxMin backward (CUDA)");
}
