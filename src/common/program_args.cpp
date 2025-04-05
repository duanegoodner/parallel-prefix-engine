// ----------------------------------------------------------------------------
// program_args.cpp
//
// Implements argument parsing using CLI11. Parses CLI options such as backend,
// local matrix size, seed, and matrix/grid dimensions.
// ----------------------------------------------------------------------------

#include "common/program_args.hpp"

#include <CLI/CLI.hpp>
#include <utility>

#include "mpi_prefix_sum/mpi_prefix_sum_solver.hpp"

#include "cuda_prefix_sum/cuda_prefix_sum_solver.hpp"

ProgramArgs::ProgramArgs(
    int local_n,
    int seed,
    std::string backend,
    bool verbose,
    std::vector<int> full_matrix_dim,
    std::vector<int> grid_dim,
    std::vector<int> tile_dim,
    int orig_argc,
    char **orig_argv
)
    : local_n_(local_n)
    , seed_(seed)
    , backend_(std::move(backend))
    , verbose_(verbose)
    , full_matrix_dim_(std::move(full_matrix_dim))
    , grid_dim_(std::move(grid_dim))
    , tile_dim_(std::move(tile_dim))
    , orig_argc_(orig_argc)
    , orig_argv_(orig_argv) {
  full_matrix_size_ = full_matrix_dim_[0] * full_matrix_dim_[1];
  num_tile_rows_ = grid_dim_[0];
  num_tile_cols_ = grid_dim_[1];
}

ProgramArgs ProgramArgs::Parse(int argc, char *const argv[]) {
  CLI::App app{"Distributed prefix sum runner"};

  int local_n = 2;
  int seed = 1234;
  std::string backend = "mpi";
  bool verbose = false;
  std::vector<int> full_matrix_dim = {4, 4};
  std::vector<int> grid_dim = {2, 2};
  std::vector<int> tile_dim = {2, 2};

  app.add_option("-n, --local-n", local_n, "Size of local (square) matrix")
      ->default_val("2");
  app.add_option("-s, --seed", seed, "Random seed")->default_val("1234");
  app.add_option("-b, --backend", backend, "Backend to use (mpi or cuda)")
      ->check(CLI::IsMember({"mpi", "cuda"}))
      ->default_val("mpi");
  app.add_flag("-v,--verbose", verbose, "Enable verbose output");

  app.add_option(
         "-f, --full-matrix-dim",
         full_matrix_dim,
         "Full matrix dimensions (rows cols)"
  )
      ->expected(2)
      ->default_val(std::vector<std::string>{"4", "4"});

  app.add_option("-g, --grid-size", grid_dim, "Grid dimensions (rows cols)")
      ->expected(2)
      ->default_val(std::vector<std::string>{"2", "2"});
  app.add_option("-t, --tile-dim",tile_dim, "Tile dimensions (rows cols)")
      ->expected(2)
      ->default_val(std::vector<std::string>{"2", "2"});

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    std::exit(app.exit(e));
  }

  return ProgramArgs(
      local_n,
      seed,
      backend,
      verbose,
      full_matrix_dim,
      grid_dim,
      tile_dim,
      argc,
      const_cast<char **>(argv)
  );
}

std::unique_ptr<PrefixSumSolver> ProgramArgs::MakeSolver() const {
  if (backend_ == "mpi") {
    return std::make_unique<MpiPrefixSumSolver>(*this);
  } else if (backend_ == "cuda") {
    return std::make_unique<CudaPrefixSumSolver>(*this);
  } else {
    throw std::runtime_error("Unsupported backend: " + backend_);
  }
}
