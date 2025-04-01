#include "cuda_prefix_sum/cuda_prefix_sum_solver.hpp"

#include <cuda_runtime.h> // Required for cudaStream_t

#include <iostream>

#include "cuda_prefix_sum/cuda_prefix_sum_solver.cuh"

// Ensure proper linkage between C++ and CUDA code
// void LaunchPrefixSumKernel(int* d_data, int tile_dim, cudaStream_t stream =
// 0);

CudaPrefixSumSolver::CudaPrefixSumSolver(const ProgramArgs &program_args)
    : program_args_(program_args) {}

void CudaPrefixSumSolver::Compute(std::vector<int> &local_matrix) {
  int tile_dim = program_args_.local_n();
  int total_elements = tile_dim * tile_dim;

  int *d_data = nullptr;

  // Allocate device memory
  cudaMalloc(&d_data, total_elements * sizeof(int));
  cudaMemcpy(
      d_data,
      local_matrix.data(),
      total_elements * sizeof(int),
      cudaMemcpyHostToDevice
  );

  start_time_ = std::chrono::steady_clock::now();

  // Launch kernel
  LaunchPrefixSumKernel(d_data, tile_dim);

  cudaDeviceSynchronize();
  stop_time_ = std::chrono::steady_clock::now();

  // Copy results back
  cudaMemcpy(
      local_matrix.data(),
      d_data,
      total_elements * sizeof(int),
      cudaMemcpyDeviceToHost
  );

  cudaFree(d_data);
}

void CudaPrefixSumSolver::PrintMatrix(
    const std::vector<int> &local_matrix,
    const std::string &header
) const {
  std::cout << header << "\n";
  int local_n = program_args_.local_n();
  for (int i = 0; i < local_n; ++i) {
    for (int j = 0; j < local_n; ++j) {
      std::cout << local_matrix[i * local_n + j] << "\t";
    }
    std::cout << "\n";
  }
}

void CudaPrefixSumSolver::StartTimer() {
  start_time_ = std::chrono::steady_clock::now();
}

void CudaPrefixSumSolver::StopTimer() {
  stop_time_ = std::chrono::steady_clock::now();
}

void CudaPrefixSumSolver::ReportTime() const {
  double time_ms =
      std::chrono::duration<double, std::milli>(stop_time_ - start_time_)
          .count();
  std::cout << "CUDA Execution time: " << time_ms << " ms" << std::endl;
}

const ProgramArgs &CudaPrefixSumSolver::program_args() const {
  return program_args_;
}
