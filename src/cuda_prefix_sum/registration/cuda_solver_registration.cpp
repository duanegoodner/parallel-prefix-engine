#include "common/solver_dispatch.hpp"
#include "cuda_prefix_sum/cuda_prefix_sum_solver.hpp"
#include "cuda_prefix_sum/cuda_solver_registration.hpp"
#include "cuda_prefix_sum/cuda_kernel_launcher_selector.hpp"
// #include "cuda_prefix_sum/internal/kernel_launcher_factory.hpp"


struct CudaSolverRegistrar {
  CudaSolverRegistrar() {
    std::cout << "[cuda] Registering CUDA backend...\n";
    PrefixSumSolverFactory::RegisterBackend("cuda",
      [](const ProgramArgs& args) {
        return std::make_unique<CudaPrefixSumSolver>(
            args, CreateCudaKernelLauncher(args));
      });
  }
};

void ForceCudaSolverRegistration() {
  static CudaSolverRegistrar instance;
}

