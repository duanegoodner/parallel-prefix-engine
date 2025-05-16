#pragma once
#include <memory>

#include "common/prefix_sum_solver.hpp"
#include "common/program_args.hpp"

std::unique_ptr<PrefixSumSolver> MakeSolver(ProgramArgs &program_args);

class PrefixSumSolverFactory {
public:
  using SolverBuilder =
      std::function<std::unique_ptr<PrefixSumSolver>(const ProgramArgs &)>;

  static std::unique_ptr<PrefixSumSolver> Create(const ProgramArgs &args) {
    const auto &backend = args.backend();
    auto it = GetRegistry().find(backend);
    if (it == GetRegistry().end()) {
      throw std::runtime_error("Unsupported backend: " + backend);
    }
    return it->second(args);
  }

  static void RegisterBackend(const std::string &name, SolverBuilder builder) {
    GetRegistry()[name] = std::move(builder);
  }

private:
  static std::unordered_map<std::string, SolverBuilder> &GetRegistry() {
    static std::unordered_map<std::string, SolverBuilder> registry;
    return registry;
  }
};

// std::unique_ptr<KernelLauncher> CreateCudaKernelLauncher(const ProgramArgs& args);