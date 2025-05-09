// ----------------------------------------------------------------------------
// program_args.cpp
//
// Implements ProgramArgs class.
// ----------------------------------------------------------------------------

#include "common/program_args.hpp"

#include <utility>

#include "common/logger.hpp"

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

  if (!TileAndFullMatrixHaveSameNumDims()) {
    throw std::invalid_argument("full_matrix_dim and tile_dim must have the "
                                "same number of dimensions");
  }

  // if (!(backend_ == "cuda") && !(cuda_kernel_ == "accum")) {
  //   if (!IsFullMatrixDimDivisibleByTileDim()) {
  //     throw std::invalid_argument("full_matrix_dim must be divisible by "
  //       "tile_dim unless using Cuda accum backend");
  //   }
  // }

  // std::cout << "Backend: " << backend_ << std::endl;
  // std::cout << "Kernel: " << cuda_kernel_ << std::endl;

  if ((backend_ != "cuda") || (cuda_kernel_ != "accum")) {
    if (!IsFullMatrixDimDivisibleByTileDim()) {
      throw std::invalid_argument("full_matrix dim must be divisible by "
                                  "tile_dim unless using cuda arch backend");
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
}
