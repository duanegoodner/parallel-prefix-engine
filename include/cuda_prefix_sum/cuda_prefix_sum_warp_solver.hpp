#pragma once

#include <vector>

class CudaPrefixSumWarpSolver {
public:
    // Run prefix sum on a host-side 32x32 input and return result
    std::vector<int> Run(const std::vector<int>& host_input);
};
