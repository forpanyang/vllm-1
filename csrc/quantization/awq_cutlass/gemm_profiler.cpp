#include "gemm_profiler.h"
#include "cutlass/numeric_types.h"
#include "cutlass/integer_subbyte.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"

#include <functional>
#include <iostream>
#include <tuple>
#include <unordered_map>

typedef std::tuple<int, int, int> gemm_key_t;

struct gemm_key_hash : public std::unary_function<gemm_key_t, std::size_t> {
  std::size_t operator()(const gemm_key_t &k) const {
    return std::get<0>(k) ^ std::get<1>(k) ^ std::get<2>(k);
  }
};

using Config = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;
static std::unordered_map<gemm_key_t, std::optional<Config>, gemm_key_hash>
    mProfileMap;
uintptr_t constexpr kCudaMemAlign = 128;

int8_t *alignPtr(int8_t *ptr, uintptr_t to) {
  uintptr_t addr = (uintptr_t)ptr;
  if (addr % to) {
    addr += to - addr % to;
  }
  return (int8_t *)addr;
}

int8_t *nextWorkspacePtrCommon(int8_t *ptr, uintptr_t previousWorkspaceSize,
                               const uintptr_t alignment) {
  uintptr_t addr = (uintptr_t)ptr;
  addr += previousWorkspaceSize;
  return alignPtr((int8_t *)addr, alignment);
}

int8_t *nextWorkspacePtr(int8_t *ptr, uintptr_t previousWorkspaceSize) {
  return nextWorkspacePtrCommon(ptr, previousWorkspaceSize, kCudaMemAlign);
}

void GemmPluginProfiler::allocateTmpData() {
  TLLM_CHECK_WITH_INFO(mTmpWorkspaceSizeInBytes > 0,
                       "tmpWorkspaceSizeInBytes must be larger than 0");
  const auto status = cudaMalloc(&mWorkspaceTmp, mTmpWorkspaceSizeInBytes);
  TLLM_CHECK_WITH_INFO(
      status == cudaSuccess,
      "Can't allocate tmp workspace for GEMM tactics profiling.");
}

void GemmPluginProfiler::freeTmpData() {
  if (mWorkspaceTmp == nullptr)
    return;
  const auto status = cudaFree(mWorkspaceTmp);
  TLLM_CHECK_WITH_INFO(status == cudaSuccess,
                       "Can't free tmp workspace for GEMM tactics profiling.");
  mWorkspaceTmp = nullptr;
}

inline int nextPowerOfTwo(int v) {
  --v;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return ++v;
}

size_t calculateTotalWorkspaceSize(size_t *workspaces, int count,
                                   const uintptr_t alignment = kCudaMemAlign) {
  size_t total = 0;
  for (int i = 0; i < count; i++) {
    total += workspaces[i];
    if (workspaces[i] % alignment) {
      total += alignment - (workspaces[i] % alignment);
    }
  }
  return total;
}

void GemmPluginProfiler::computeTmpSize(int maxM, int n, int k) {
  const int originalN = n * 8;
  std::vector<size_t> workspaces = {
      maxM * k * sizeof(half),                   // A
      k * n * sizeof(float),                     // B
      k * originalN * sizeof(half) / mGroupSize, // scales
      k * originalN * sizeof(half) / mGroupSize, // zeros
      maxM * sizeof(half),                       // biases
      maxM * originalN * sizeof(half),           // C
      mRunner->getWorkspaceSize(maxM, n, k)      // workspace
  };
  size_t bytes =
      calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
  setTmpWorkspaceSizeInBytes(bytes);
}

static constexpr int MAX_PROFILE_M = 8192;
void GemmPluginProfiler::profileTactics(const WeightOnlyGemmRunnerPtr &runner,
                                        int n, int k, int maxM) {
  mRunner = runner;
  maxM = std::min(nextPowerOfTwo(maxM), MAX_PROFILE_M);
  computeTmpSize(maxM, n, k);

  auto profileTactics = [&](int m, int n, int k) {
    if (mProfileMap.count({m, n, k}) == 0) {
      const auto tactics = this->mRunner->getConfigs();
      // Profile different tactics for particular m and insert best config to
      // the map
      mProfileMap.insert({std::make_tuple(m, n, k),
                          this->profileTacticsForProblem(m, n, k, tactics)});
    }
  };

  // Allocate tmp data to run GEMMs
  allocateTmpData();
  int minM = 32;
  const int startMinMRounded = nextPowerOfTwo(minM);
  for (int m = startMinMRounded; m < maxM; m += 32) {
    profileTactics(m, n, k);
  }

  profileTactics(maxM, n, k);
  // Free tmp data
  freeTmpData();
}

std::optional<Config> GemmPluginProfiler::getBestConfig(int m, int n,
                                                        int k) const {
  int mRounded = (m + 32 - 1) / 32 * 32;
  mRounded = std::min(mRounded, MAX_PROFILE_M);
  // const int mRounded = std::min(nextPowerOfTwo(m), MAX_PROFILE_M);
  if (mProfileMap.count({mRounded, n, k}) == 0) {
    return std::nullopt;
  }
  return mProfileMap.at({mRounded, n, k});
}

std::optional<Config> GemmPluginProfiler::profileTacticsForProblem(
    int m, int n, int k, const std::vector<Config> &tactics) {
  TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

  float bestTime = std::numeric_limits<float>::max();
  Config bestConfig;
  bool foundOne = false;

  // Iterate over all tactics for given M, N and K
  for (int ii = 0; ii < tactics.size(); ++ii) {
    const Config &candidateConfig = tactics[ii];
    float time = std::numeric_limits<float>::max();
    try {
      // Profile particualar tactic for given M, N and K
      time = profileTacticForProblem(m, n, k, candidateConfig);
      foundOne = true;
    } catch (const std::exception &e) {
      std::ostringstream msg;
      msg << "Cannot profile configuration " << ii << " (for"
          << " m=" << m << ", n=" << n << ", k=" << k << "). Skipped"
          << ". " << e.what();
      TLLM_LOG_WARNING(msg.str());
      continue;
    }

    // Choose the fastest tactic
    if (time < bestTime) {
      bestConfig = candidateConfig;
      bestTime = time;
      std::ostringstream msg;
      msg << "Update best profile :" << ii << " (for"
          << " m=" << m << ", n=" << n << ", k=" << k << "). "
          << "bestTime: " << bestTime;
      std::cout << msg.str() << std::endl;
    }
  }

  if (!foundOne) {
    std::ostringstream msg;
    msg << "Have not found any valid GEMM config for shape ("
        << "m=" << m << ", n=" << n << ", k=" << k
        << "). Will try to use default or fail at runtime";
    TLLM_LOG_WARNING(msg.str());
    return std::nullopt;
  }
  return {bestConfig};
}

float GemmPluginProfiler::profileTacticForProblem(int m, int n, int k,
                                                  const Config &tactic) {
  constexpr int warmup = 5;
  constexpr int runs = 10;

  cudaStream_t stream = cudaStreamDefault;
  // Warmup the execution
  for (int i = 0; i < warmup; ++i) {
    runTactic(m, n, k, tactic, mWorkspaceTmp, stream);
  }

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaDeviceSynchronize();
  cudaEventRecord(start, 0);

  // Profile GEMM
  for (int i = 0; i < runs; ++i) {
    runTactic(m, n, k, tactic, mWorkspaceTmp, stream);
  }

  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);

  float elapsed;
  cudaEventElapsedTime(&elapsed, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return elapsed / runs;
}

void GemmPluginProfiler::runTactic(int m, int n, int k, const Config &tactic,
                                   void *workspace,
                                   const cudaStream_t &stream) {
  const int originalN = n * 8;
  half *actPtr = reinterpret_cast<half *>(workspace);
  cutlass::uint4b_t *weightPtr =
      reinterpret_cast<cutlass::uint4b_t *>(nextWorkspacePtr(
          reinterpret_cast<int8_t *>(actPtr), m * k * sizeof(half)));
  half *inputScalesPtr = reinterpret_cast<half *>(nextWorkspacePtr(
      reinterpret_cast<int8_t *>(weightPtr), n * k * sizeof(float)));
  half *zerosPtr = reinterpret_cast<half *>(
      nextWorkspacePtr(reinterpret_cast<int8_t *>(inputScalesPtr),
                       k * originalN * sizeof(half) / mGroupSize));
  half *outputPtr = reinterpret_cast<half *>(
      nextWorkspacePtr(reinterpret_cast<int8_t *>(zerosPtr),
                       k * originalN * sizeof(half) / mGroupSize));
  char *workspacePtr = reinterpret_cast<char *>(nextWorkspacePtr(
      reinterpret_cast<int8_t *>(outputPtr), m * originalN * sizeof(half)));
  half *biasesPtr = nullptr;
  const int wsSize = mRunner->getWorkspaceSize(m, n, k);

  mRunner->gemm(actPtr, weightPtr, inputScalesPtr, zerosPtr, biasesPtr,
                outputPtr, m, originalN, k, mGroupSize, tactic, workspacePtr,
                wsSize, stream);
}
