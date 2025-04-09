#include "cuda_prefix_sum/cuda_prefix_sum_solver.hpp"

#include <cuda_runtime.h> // Required for cudaStream_t

#include <iostream>

#include "common/matrix_init.hpp"

#include "cuda_prefix_sum/cuda_prefix_sum_solver.cuh"

#include "cuda_prefix_sum/kernel_launch_params.hpp"

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

void CudaPrefixSumSolver::Compute() {
  int *d_data = nullptr;

  // Allocate device memory
  cudaMalloc(&d_data, program_args_.FullMatrixSize() * sizeof(int));
  cudaMemcpy(
      d_data,
      full_matrix_.data(),
      program_args_.FullMatrixSize() * sizeof(int),
      cudaMemcpyHostToDevice
  );

  start_time_ = std::chrono::steady_clock::now();

  auto launch_params = CreateKernelLaunchParams(d_data, program_args_);

  // Launch kernel
  LaunchPrefixSumKernel(
      d_data,
      launch_params,
      0 // Use the default CUDA stream
  );

  cudaDeviceSynchronize();

  // Copy results back
  cudaMemcpy(
      full_matrix_.data(),
      d_data,
      program_args_.FullMatrixSize() * sizeof(int),
      cudaMemcpyDeviceToHost
  );

  cudaFree(d_data);
}

void CudaPrefixSumSolver::PrintFullMatrix(std::string title) {
  std::cout << title << std::endl;
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
  end_time_ = std::chrono::steady_clock::now();
}

std::chrono::duration<double> CudaPrefixSumSolver::GetElapsedTime() const {
  return end_time_ - start_time_;
}

std::chrono::duration<double> CudaPrefixSumSolver::GetStartTime() const {
  return std::chrono::duration<double>(start_time_.time_since_epoch());
}

std::chrono::duration<double> CudaPrefixSumSolver::GetEndTime() const {
  return std::chrono::duration<double>(end_time_.time_since_epoch());
}

void CudaPrefixSumSolver::ReportTime() const {
  double elapsed_time_s = GetElapsedTime().count();
  std::cout << "CUDA Execution time: " << elapsed_time_s * 1000 << " ms"
            << std::endl;
}

const ProgramArgs &CudaPrefixSumSolver::program_args() const {
  return program_args_;
}
