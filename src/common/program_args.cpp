// ----------------------------------------------------------------------------
// program_args.cpp
//
// Implements argument parsing using CLI11. Parses CLI options such as backend,
// local matrix size, seed, and matrix/grid dimensions.
// ----------------------------------------------------------------------------

#include "common/program_args.hpp"

// #include <CLI/CLI.hpp>
#include <utility>

// #include "mpi_prefix_sum/mpi_prefix_sum_solver.hpp"

// #include "cuda_prefix_sum/cuda_prefix_sum_solver.hpp"

ProgramArgs::ProgramArgs(
    int seed,
    std::string backend,
    bool verbose,
    std::vector<int> full_matrix_dim,
    std::vector<int> tile_dim,
    int orig_argc,
    char **orig_argv
)
    : seed_(seed)
    , backend_(std::move(backend))
    , verbose_(verbose)
    , full_matrix_dim_(std::move(full_matrix_dim))
    , tile_dim_(std::move(tile_dim))
    , orig_argc_(orig_argc)
    , orig_argv_(orig_argv) {
  if (full_matrix_dim_.size() != tile_dim_.size()) {
    throw std::invalid_argument(
        "full_matrix_dim and tile_dim must have the same number of dimensions"
    );
  }

  for (size_t i = 0; i < full_matrix_dim_.size(); ++i) {
    if (full_matrix_dim_[i] % tile_dim_[i] != 0) {
      throw std::invalid_argument(
          "full_matrix_dim[" + std::to_string(i) +
          "] is not divisible by tile_dim[" + std::to_string(i) + "]"
      );
    }
  }
}
