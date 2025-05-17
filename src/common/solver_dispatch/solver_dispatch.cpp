#include "common/solver_dispatch.hpp"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "common/program_args.hpp"

#include "mpi_prefix_sum/mpi_prefix_sum_solver.hpp"

#include "cuda_prefix_sum/cuda_prefix_sum_solver.hpp"
#include "cuda_prefix_sum/kernel_launcher.hpp"
#include "cuda_prefix_sum/cuda_kernel_launcher_selector.hpp"
#include "cuda_prefix_sum/multi_tile_kernel_launcher.cuh"
#include "cuda_prefix_sum/single_tile_kernel_launcher.cuh"
#include "cuda_prefix_sum/cuda_solver_registration.hpp"
#include "mpi_prefix_sum/mpi_solver_registration.hpp"

std::unique_ptr<PrefixSumSolver> MakeSolver(ProgramArgs &program_args) {
  static const std::unordered_map<
      std::string,
      std::function<std::unique_ptr<PrefixSumSolver>(ProgramArgs &)>>
      solver_factories = {
          {"mpi",
           [](ProgramArgs &args) {
             return std::make_unique<MpiPrefixSumSolver>(args);
           }},
          {"cuda", [](ProgramArgs &args) {
             using LauncherCreator =
                 std::function<std::unique_ptr<KernelLauncher>()>;

             static const std::unordered_map<std::string, LauncherCreator>
                 kernel_map = {
                     {"single_tile",
                      [args] {
                        return std::make_unique<SingleTileKernelLauncher>(args);
                      }},
                     {"multi_tile",
                      [args] {
                        return std::make_unique<MultiTileKernelLauncher>(args);
                      }},
                     // Add more kernels here
                 };

             if (!args.cuda_kernel().has_value()) {
               throw std::runtime_error("Missing CUDA kernel name: --kernel "
                                        "is required with --backend=cuda");
             }

             const std::string &kernel = args.cuda_kernel().value();

             auto it = kernel_map.find(kernel);
             if (it == kernel_map.end()) {
               throw std::runtime_error("Unsupported CUDA kernel: " + kernel);
             }

             return std::make_unique<CudaPrefixSumSolver>(args, it->second());
           }}};

  auto it = solver_factories.find(program_args.backend());
  if (it != solver_factories.end()) {
    return it->second(program_args);
  } else {
    throw std::runtime_error("Unsupported backend: " + program_args.backend());
  }
}



void RegisterAllSolvers() {
  ForceCudaSolverRegistration();
  ForceMpiSolverRegistration();
}

// Registration
static bool registered = [] {
    PrefixSumSolverFactory::RegisterBackend("mpi",
        [](const ProgramArgs& args) {
            return std::make_unique<MpiPrefixSumSolver>(args);
        });

    PrefixSumSolverFactory::RegisterBackend("cuda",
        [](const ProgramArgs& args) {
            return std::make_unique<CudaPrefixSumSolver>(args, CreateCudaKernelLauncher(args));
        });

    return true;
}();

