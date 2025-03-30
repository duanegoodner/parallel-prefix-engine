#pragma once

#include "mpi_prefix_sum/mpi_environment.hpp"
#include "mpi_prefix_sum/mpi_utils.hpp"
#include "mpi_prefix_sum/prefix_sum_block_matrix.hpp"
#include <vector>

void MyPrefixSum(
    const MpiEnvironment &mpi,
    const ProgramArgs &args,
    std::vector<int> &sum_matrix
);
