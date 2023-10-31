#include <torch/extension.h>

#include <map>
#include <vector>

void swap_blocks(
  torch::Tensor& src,
  torch::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping);

void copy_blocks(
  std::vector<torch::Tensor>& key_caches,
  std::vector<torch::Tensor>& value_caches,
  const std::map<int64_t, std::vector<int64_t>>& block_mapping);

void copy_blocks_quantized(
    std::vector<torch::Tensor> &key_caches,
    std::vector<torch::Tensor> &value_caches,
    std::vector<torch::Tensor> &quant_params,
    const std::map<int64_t, std::vector<int64_t>> &block_mapping);

void reshape_and_cache(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& slot_mapping);

void reshape_and_cache_fp8(torch::Tensor &key, torch::Tensor &value,
                           torch::Tensor &key_cache, torch::Tensor &value_cache,
                           torch::Tensor &slot_mapping);

void reshape_and_cache_quantized(torch::Tensor &key, torch::Tensor &value,
                                 torch::Tensor &q_params,
                                 torch::Tensor &key_cache,
                                 torch::Tensor &value_cache,
                                 torch::Tensor &q_params_cache,
                                 torch::Tensor &slot_mapping);

void gather_cached_kv(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& slot_mapping);

void gather_cached_kv_fp8(torch::Tensor &key, torch::Tensor &value,
                          torch::Tensor &key_cache, torch::Tensor &value_cache,
                          torch::Tensor &slot_mapping);

void gather_cached_kv_quantized(torch::Tensor &key, torch::Tensor &value,
                                torch::Tensor &q_param,
                                torch::Tensor &key_cache,
                                torch::Tensor &value_cache,
                                torch::Tensor &q_param_cache,
                                torch::Tensor &slot_mapping);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "swap_blocks",
    &swap_blocks,
    "Swap in (out) the cache blocks from src to dst");
  m.def(
    "copy_blocks",
    &copy_blocks,
    "Copy the cache blocks from src to dst");
  m.def("copy_blocks_quantized", &copy_blocks_quantized,
        "Copy the quantized cache blocks from src to dst");
  m.def(
    "reshape_and_cache",
    &reshape_and_cache,
    "Reshape the key and value tensors and cache them");
  m.def("reshape_and_cache_quantized", &reshape_and_cache_quantized,
        "Reshape the quantized key and value tensors and cache them");
  m.def("reshape_and_cache_fp8", &reshape_and_cache_fp8,
        "Reshape the key and value tensors and cache them(fp8)");
  m.def(
    "gather_cached_kv",
    &gather_cached_kv,
    "Gather key and value from the cache into contiguous QKV tensors");
  m.def("gather_cached_kv_quantized", &gather_cached_kv_quantized,
        "Gather key and value from the quantized cache into contiguous QKV "
        "tensors");
  m.def("gather_cached_kv_fp8", &gather_cached_kv_fp8,
        "Gather key and value from the quantized cache into contiguous QKV "
        "tensors");
}
