#include "cuda_prefix_sum/cuda_prefix_sum_warp_solver.hpp"
#include "cuda_prefix_sum/cuda_prefix_sum_solver.cuh"

#include <cuda_runtime.h>
#include <stdexcept>

namespace config {
    constexpr int TileDim = 32;
    constexpr int NumElems = TileDim * TileDim;
}

std::vector<int> CudaPrefixSumWarpSolver::Run(const std::vector<int>& host_input) {
    if (host_input.size() != config::NumElems) {
        throw std::invalid_argument("Input size must be 32x32 (1024 elements)");
    }

    std::vector<int> host_output(config::NumElems);

    int* d_input = nullptr;
    int* d_output = nullptr;

    cudaMalloc(&d_input, config::NumElems * sizeof(int));
    cudaMalloc(&d_output, config::NumElems * sizeof(int));

    cudaMemcpy(d_input, host_input.data(), config::NumElems * sizeof(int), cudaMemcpyHostToDevice);

    LaunchPrefixSumKernelWarp(d_input, d_output);

    cudaMemcpy(host_output.data(), d_output, config::NumElems * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return host_output;
}
