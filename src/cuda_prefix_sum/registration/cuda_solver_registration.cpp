#include "common/solver_dispatch.hpp"
#include "cuda_prefix_sum/cuda_prefix_sum_solver.hpp"
// #include "cuda_prefix_sum/internal/kernel_launcher_factory.hpp"


struct CudaSolverRegistrar {
  CudaSolverRegistrar() {
    PrefixSumSolverFactory::RegisterBackend("cuda",
      [](const ProgramArgs& args) {
        return std::make_unique<CudaPrefixSumSolver>(
            args, CreateCudaKernelLauncher(args));
      });
  }
};

const CudaSolverRegistrar& GetCudaSolverRegistrar() {
  static CudaSolverRegistrar instance;
  return instance;
}

