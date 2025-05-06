#include "common/solver_dispatch.hpp"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "common/program_args.hpp"

#include "mpi_prefix_sum/mpi_prefix_sum_solver.hpp"

#include "cuda_prefix_sum/cuda_prefix_sum_solver.cuh"
#include "cuda_prefix_sum/cuda_prefix_sum_solver.hpp"

std::unique_ptr<PrefixSumSolver> MakeSolver(ProgramArgs &program_args) {
  // Define a map of backend strings to factory functions
  static const std::unordered_map<
      std::string,
      std::function<std::unique_ptr<PrefixSumSolver>(ProgramArgs &)>>
      solver_factories = {
          {"mpi",
           [](ProgramArgs &args) {
             return std::make_unique<MpiPrefixSumSolver>(args);
           }},
          {"cuda", [](ProgramArgs &args) {
             using KLF = CudaPrefixSumSolver::KernelLaunchFunction;

             // Map string tag to function pointer
             static const std::unordered_map<std::string, KLF> kernel_map = {
                 {"tiled", LaunchPrefixSumKernelTiled},
                 {"single_element", LaunchPrefixSumKernelSingleElement},
                //  {"warp", LaunchPrefixSumKernelWarp},
                //  {"warp_naive", LaunchPrefixSumKernelWarpNaive},
                 {"accum", LaunchPrefixSumKernelAccum},
                //  {"hybrid", LaunchPrefixSumKernelHybrid}
                //  {"arch", LaunchPrefixSumKernelHierarchical}
             };

             auto it = kernel_map.find(args.cuda_kernel());
             if (it == kernel_map.end()) {
               throw std::runtime_error(
                   "Unsupported CUDA kernel: " + args.cuda_kernel()
               );
             }
             return std::make_unique<CudaPrefixSumSolver>(
                 args,
                 it->second
             );
           }}};

  // Find the factory function for the requested backend
  auto it = solver_factories.find(program_args.backend());
  if (it != solver_factories.end()) {
    return it->second(program_args); // Call the factory function
  } else {
    throw std::runtime_error("Unsupported backend: " + program_args.backend());
  }
}