#include "cuda_prefix_sum/cuda_prefix_sum_solver.hpp"

#include <cuda_runtime.h> // Required for cudaStream_t

#include <iostream>

#include "common/matrix_init.hpp"

#include "cuda_prefix_sum/cuda_prefix_sum_solver.cuh"

// Ensure proper linkage between C++ and CUDA code
// void LaunchPrefixSumKernel(int* d_data, int tile_dim, cudaStream_t stream =
// 0);

CudaPrefixSumSolver::CudaPrefixSumSolver(const ProgramArgs &program_args)
    : program_args_(program_args) {
      PopulateFullMatrix();
    }

void CudaPrefixSumSolver::PopulateFullMatrix() {
  full_matrix_ = GenerateRandomMatrix<int>(
      program_args_.full_matrix_dim()[0],
      program_args_.full_matrix_dim()[1],
      program_args_.seed()
  );
}


void CudaPrefixSumSolver::ComputeNew() {
  // This function is not used in the current implementation
  // It can be removed or implemented as needed

  int *d_data = nullptr;

  cudaMalloc(&d_data, program_args_.FullMatrixSize() * sizeof(int));
  cudaMemcpy(
      d_data,
      full_matrix_.data(),
      program_args_.FullMatrixSize() * sizeof(int),
      cudaMemcpyHostToDevice
  );

  start_time_ = std::chrono::steady_clock::now();

  // Launch kernel
  LaunchPrefixSumKernel(
    d_data,
    program_args_.tile_dim()[0],
    program_args_.tile_dim()[1],
    program_args_.GridDim()[0],
    program_args_.GridDim()[1],
    0 // Use the default CUDA stream
);

cudaDeviceSynchronize();
stop_time_ = std::chrono::steady_clock::now();

// Copy results back
cudaMemcpy(
  full_matrix_.data(),
  d_data,
  program_args_.ElementsPerTile() * sizeof(int),
  cudaMemcpyDeviceToHost
);

cudaFree(d_data);
}



void CudaPrefixSumSolver::Compute(std::vector<int> &local_matrix) {

  int *d_data = nullptr;

  // Allocate device memory
  cudaMalloc(&d_data, program_args_.ElementsPerTile() * sizeof(int));
  cudaMemcpy(
      d_data,
      local_matrix.data(),
      program_args_.ElementsPerTile() * sizeof(int),
      cudaMemcpyHostToDevice
  );

  start_time_ = std::chrono::steady_clock::now();

  // Launch kernel
  LaunchPrefixSumKernel(
      d_data,
      program_args_.tile_dim()[0],
      program_args_.tile_dim()[1],
      program_args_.GridDim()[0],
      program_args_.GridDim()[1],
      0 // Use the default CUDA stream
  );

  cudaDeviceSynchronize();
  stop_time_ = std::chrono::steady_clock::now();

  // Copy results back
  cudaMemcpy(
      local_matrix.data(),
      d_data,
      program_args_.ElementsPerTile() * sizeof(int),
      cudaMemcpyDeviceToHost
  );

  cudaFree(d_data);
}

void CudaPrefixSumSolver::PrintMatrix(
    const std::vector<int> &local_matrix,
    const std::string &header
) const {
  std::cout << header << "\n";
  int num_rows = program_args_.tile_dim()[0];
  int num_cols = program_args_.tile_dim()[1];
  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_cols; ++j) {
      std::cout << local_matrix[i * num_cols + j] << "\t";
    }
    std::cout << "\n";
  }
}

void CudaPrefixSumSolver::PrintFullMatrix() {
  for (auto row = 0; row < program_args_.full_matrix_dim()[0]; ++row) {
    for (auto col = 0; col < program_args_.full_matrix_dim()[1]; ++col) {
      std::cout << full_matrix_[row * program_args_.full_matrix_dim()[1] + col]
                << "\t";
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
