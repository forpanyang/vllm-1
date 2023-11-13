#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"

using WeightOnlyGemmRunner =
    tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface;
using WeightOnlyGemmRunnerPtr = std::shared_ptr<WeightOnlyGemmRunner>;

class GemmPluginProfiler {
public:
  using Config = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;
  void setGroupSize(int groupSize) { mGroupSize = groupSize; }
  void setTmpWorkspaceSizeInBytes(size_t bytes) {
    mTmpWorkspaceSizeInBytes = bytes;
  }
  void profileTactics(const WeightOnlyGemmRunnerPtr &runner, int n, int k,
                      int maxM);
  void computeTmpSize(int maxM, int n, int k);

  std::optional<Config>
  profileTacticsForProblem(int m, int n, int k,
                           const std::vector<Config> &tactics);
  float profileTacticForProblem(int m, int n, int k, const Config &tactic);

  void allocateTmpData();
  void freeTmpData();
  std::optional<Config> getBestConfig(int m, int n, int k) const;

protected:
  void runTactic(int m, int n, int k, const Config &tactic, void *workspace,
                 const cudaStream_t &stream);

private:
  int mGroupSize;
  void *mWorkspaceTmp{nullptr};
  size_t mTmpWorkspaceSizeInBytes{0};
  WeightOnlyGemmRunnerPtr mRunner{nullptr};
};
