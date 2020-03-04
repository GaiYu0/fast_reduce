#include <torch/extension.h>

#define CHECK_CUDA(x) 
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor min2d_2nd_dim_cuda_forward(torch::Tensor x);

torch::Tensor min2d_2nd_dim_forward(torch::Tensor x) {
  TORCH_CHECK(x.type().is_cuda(), "x must be a CUDA tensor!");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous!");
  TORCH_CHECK(x.dim() == 2, "x must be 2D!");
  return min2d_2nd_dim_cuda_forward(x);
}

torch::Tensor min2d_2nd_dim_backward(torch::Tensor x) {
  TORCH_CHECK(false, "Not implemented!");
  return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &min2d_2nd_dim_forward, "MIN2D_2ND_DIM forward (CUDA)");
  m.def("backward", &min2d_2nd_dim_backward, "MIN2D_2ND_DIM backward (CUDA)");
}
