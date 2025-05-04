// ----------------------------------------------------------------------------
// program_args.cpp
//
// Implements ProgramArgs class.
// ----------------------------------------------------------------------------

#include "common/program_args.hpp"
#include "common/logger.hpp"

#include <utility>

ProgramArgs::ProgramArgs(
    int seed,
    std::string backend,
    LogLevel log_level,
    std::vector<int> full_matrix_dim,
    std::vector<int> tile_dim,
    std::string cuda_kernel,
    int orig_argc,
    char **orig_argv
)
    : seed_(seed)
    , backend_(std::move(backend))
    , log_level_(log_level)
    , full_matrix_dim_(std::move(full_matrix_dim))
    , tile_dim_(std::move(tile_dim))
    , cuda_kernel_(cuda_kernel)
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
