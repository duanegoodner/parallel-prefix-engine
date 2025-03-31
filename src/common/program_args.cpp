// ----------------------------------------------------------------------------
// program_args.cpp
//
// Implements argument parsing using CLI11. Parses CLI options such as backend,
// local matrix size, and optional seed value.
// ----------------------------------------------------------------------------

#include "common/program_args.hpp"

#include <CLI/CLI.hpp>
#include <utility> // for std::move

#include "mpi_prefix_sum/mpi_prefix_sum_solver.hpp"

#include "cuda_prefix_sum/cuda_prefix_sum_solver.hpp"

ProgramArgs::ProgramArgs(
    int local_n,
    int seed,
    std::string backend,
    bool verbose
)
    : local_n_(local_n)
    , seed_(seed)
    , backend_(std::move(backend))
    , verbose_(verbose) {}

ProgramArgs ProgramArgs::Parse(int argc, char *const argv[]) {
  CLI::App app{"Distributed prefix sum runner"};

  int local_n;
  int seed = 1234;
  std::string backend = "mpi";

  bool verbose = false;

  app.add_flag(
      "-v,--verbose",
      verbose,
      "Enable verbose output of parsed arguments"
  );
  app.add_option("local_n", local_n, "Size of local matrix")->required();
  app.add_option("seed", seed, "Optional seed");
  app.add_option(
         "--backend",
         backend,
         "Prefix sum backend to use (one of: mpi, cuda)"
  )
      ->check(CLI::IsMember({"mpi", "cuda"}))
      ->default_val("mpi");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    std::exit(app.exit(e)); // will print error and help message
  }

  return ProgramArgs(local_n, seed, backend, verbose);
}

std::unique_ptr<PrefixSumSolver> ProgramArgs::MakeSolver(
    int argc,
    char *argv[]
) const {
  if (backend_ == "mpi") {
    return std::make_unique<MpiPrefixSumSolver>(argc, argv);
  } else if (backend_ == "cuda") {
    return std::make_unique<CudaPrefixSumSolver>(argc, argv);
  } else {
    throw std::runtime_error("Unsupported backend: " + backend_);
  }
}
