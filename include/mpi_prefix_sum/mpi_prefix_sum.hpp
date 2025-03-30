#pragma once

#include "mpi_prefix_sum/prefix_sum_block_matrix.hpp"
#include <vector>


void MyPrefixSum(int local_n, std::vector<int>& sum_matrix);