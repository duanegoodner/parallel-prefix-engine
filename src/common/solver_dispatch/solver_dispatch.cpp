#include "common/solver_dispatch.hpp"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "common/program_args.hpp"

#include "mpi_prefix_sum/mpi_prefix_sum_solver.hpp"
#include "mpi_prefix_sum/mpi_solver_registration.hpp"

#include "cuda_prefix_sum/cuda_kernel_launcher_selector.hpp"
#include "cuda_prefix_sum/cuda_prefix_sum_solver.hpp"
#include "cuda_prefix_sum/cuda_solver_registration.hpp"
#include "cuda_prefix_sum/kernel_launcher.hpp"
#include "cuda_prefix_sum/multi_tile_kernel_launcher.cuh"
#include "cuda_prefix_sum/single_tile_kernel_launcher.cuh"

void RegisterAllSolvers() {
  ForceCudaSolverRegistration();
  ForceMpiSolverRegistration();
}

// Registration
static bool registered = [] {
  PrefixSumSolverFactory::RegisterBackend("mpi", [](const ProgramArgs &args) {
    return std::make_unique<MpiPrefixSumSolver>(args);
  });

  PrefixSumSolverFactory::RegisterBackend("cuda", [](const ProgramArgs &args) {
    return std::make_unique<CudaPrefixSumSolver>(
        args,
        CreateCudaKernelLauncher(args)
    );
  });

  return true;
}();
