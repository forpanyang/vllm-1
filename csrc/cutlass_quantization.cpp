#include <cstdint>
#include <torch/extension.h>

torch::Tensor awq_gemm_cutlass(
  torch::Tensor _in_feats,
  torch::Tensor _kernel,
  torch::Tensor _scaling_factors,
  torch::Tensor _zeros);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "awq_gemm_cutlass",
    &awq_gemm_cutlass,
    "Quantized GEMM for AWQ");
}
