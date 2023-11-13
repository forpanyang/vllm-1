#include "gemm_profiler.h"

#include <ATen/cuda/CUDAContext.h>
#include <cassert>
#include <memory>
#include <set>
#include <string>
#include <torch/extension.h>
#include <vector>

#include "cutlass/numeric_types.h"
#include "cutlass/integer_subbyte.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
// #include "tensorrt_llm/kernels/weightOnlyBatchedGemv/enabled.h"
// #include "tensorrt_llm/kernels/weightOnlyBatchedGemv/kernelLauncher.h"

namespace tkc = tensorrt_llm::cutlass_extensions;
static constexpr int SMALL_M_FAST_PATH = 5;
using WeightOnlyGemmRunner =
    tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface;
using WeightOnlyGemmRunnerPtr = std::shared_ptr<WeightOnlyGemmRunner>;

static bool profiled = false;
torch::Tensor awq_gemm_cutlass(torch::Tensor _in_feats,
                                   torch::Tensor _kernel,
                                   torch::Tensor _scaling_factors,
                                   torch::Tensor _zeros) {
  // inputs
  //   activations      [M, K]
  //   weights          [K, N/8]
  //   scales           [K // group_size, N]
  //   zeros            [K // group_size, N]

  int m = _in_feats.size(0);
  int k = _in_feats.size(1);
  int n = _kernel.size(1);
  auto options = torch::TensorOptions()
                     .dtype(_in_feats.dtype())
                     .device(_in_feats.device());
  at::Tensor _out_feats = torch::empty({m, n * 8}, options);

  auto in_feats = reinterpret_cast<half *>(_in_feats.data_ptr<at::Half>());
  auto kernel = reinterpret_cast<cutlass::uint4b_t *>(_kernel.data_ptr());
  auto out_feats = reinterpret_cast<half *>(_out_feats.data_ptr<at::Half>());
  auto scaling_factors =
      reinterpret_cast<half *>(_scaling_factors.data_ptr<at::Half>());
  auto zeros = reinterpret_cast<half *>(_zeros.data_ptr<at::Half>());
  int group_size = k / _scaling_factors.size(0);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // gemm_config.tile_config =
  // tkc::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64;
  // gemm_config.stages = 4;

  WeightOnlyGemmRunnerPtr m_weightOnlyGroupwiseGemmRunner = std::make_shared<
      tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<
          half, cutlass::uint4b_t,
          cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();

  auto configs = m_weightOnlyGroupwiseGemmRunner->getConfigs();
  // std::cout << "Num configs: " << configs.size() << std::endl;
  // tkc::CutlassGemmConfig gemm_config = configs[0];
  // GemmPluginProfiler profiler;
  // profiler.setGroupSize(group_size);
  // // config gemm
  tkc::CutlassGemmConfig gemm_config = configs[0];
  // if (not profiled or not profiler.getBestConfig(m, n, k).has_value()) {
  //   profiler.profileTactics(m_weightOnlyGroupwiseGemmRunner, n, k, 2048);
  //   profiled = true;
  //   gemm_config = profiler.getBestConfig(m, n, k).value();
  // } else {
  //   gemm_config = profiler.getBestConfig(m, n, k).value();
  // }
  // gemm_config = configs[0];
  //    if (m < SMALL_M_FAST_PATH)
  //  {
  //      // Use CUDA kernels for small batch size
  //      // The CUDA kernel is designed for ColumnMajorTileInterleave weight layout used in fpAIntB cutlass kernel
  //      // when sm >= 75 and the preprocessing of cutlass on sm70 does not interleave the weights.
  //      tensorrt_llm::kernels::WeightOnlyParams params{
  //          reinterpret_cast<const uint8_t*>(kernel),
  //          scaling_factors, zeros, in_feats, nullptr/*biases*/,
  //          out_feats, m, n * 8, k, group_size};
  //      tensorrt_llm::kernels::weight_only_batched_gemv_launcher(tensorrt_llm::kernels::WeightOnlyQuantType::Int4b,
  //          tensorrt_llm::kernels::WeightOnlyType::GroupWise,
  //          tensorrt_llm::kernels::WeightOnlyActivationType::Identity, params, stream);
  //  } else {
       const int ws_bytes =
           m_weightOnlyGroupwiseGemmRunner->getWorkspaceSize(m, n, k);
       // to avoid error
       char *workspace =
           reinterpret_cast<char *>(torch::empty({ws_bytes}, options).data_ptr());

      m_weightOnlyGroupwiseGemmRunner->gemm(
      in_feats, kernel, scaling_factors, zeros, nullptr /*biases*/, out_feats,
      m, n * 8, k, group_size, gemm_config,
      // nullptr, 0,
      reinterpret_cast<char *>(workspace) + m * k * sizeof(half), ws_bytes,
      stream);
   // }

  return _out_feats;
}
