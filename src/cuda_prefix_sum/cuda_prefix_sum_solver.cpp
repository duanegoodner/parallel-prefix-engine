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

  StartCopyToDeviceTimer();

  // Allocate device memory
  cudaMalloc(&d_data, program_args_.FullMatrixSize() * sizeof(int));
  cudaMemcpy(
      d_data,
      full_matrix_.data(),
      program_args_.FullMatrixSize() * sizeof(int),
      cudaMemcpyHostToDevice
  );

  StopCopyToDeviceTimer();

  StartDeviceComputeTimer();

  auto launch_params = CreateKernelLaunchParams(d_data, program_args_);

  // Launch kernel
  LaunchPrefixSumKernel(launch_params, 0);
  // LaunchPrefixSumKernelSingleElement(launch_params, 0);

  cudaDeviceSynchronize();

  StopDeviceComputeTimer();

  StartCopyFromDeviceTimer();

  // Copy results back
  cudaMemcpy(
      full_matrix_.data(),
      d_data,
      program_args_.FullMatrixSize() * sizeof(int),
      cudaMemcpyDeviceToHost
  );

  cudaFree(d_data);

  StopCopyFromDeviceTimer();
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

void CudaPrefixSumSolver::StartCopyToDeviceTimer() {
  copy_to_device_start_time_ = std::chrono::steady_clock::now();
}

void CudaPrefixSumSolver::StopCopyToDeviceTimer() {
  copy_to_device_end_time_ = std::chrono::steady_clock::now();
}

void CudaPrefixSumSolver::StartDeviceComputeTimer() {
  device_compute_start_time_ = std::chrono::steady_clock::now();
}

void CudaPrefixSumSolver::StopDeviceComputeTimer() {
  device_compute_end_time_ = std::chrono::steady_clock::now();
}

void CudaPrefixSumSolver::StartCopyFromDeviceTimer() {
  copy_from_device_start_time_ = std::chrono::steady_clock::now();
}

void CudaPrefixSumSolver::StopCopyFromDeviceTimer() {
  copy_from_device_end_time_ = std::chrono::steady_clock::now();
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

std::chrono::duration<double> CudaPrefixSumSolver::GetCopyToDeviceTime(
) const {
  return copy_to_device_end_time_ - copy_to_device_start_time_;
}

std::chrono::duration<double> CudaPrefixSumSolver::GetDeviceComputeTime(
) const {
  return device_compute_end_time_ - device_compute_start_time_;
}

std::chrono::duration<double> CudaPrefixSumSolver::GetCopyFromDeviceTime(
) const {
  return copy_from_device_end_time_ - copy_from_device_start_time_;
}

void CudaPrefixSumSolver::ReportTime() const {
  double elapsed_time_s = GetElapsedTime().count();
  std::cout << "\n=== Runtime Report ===\n";
  std::cout << "Total Time: " << elapsed_time_s * 1000 << " ms"
            << std::endl;

  double copy_to_device_time_s = GetCopyToDeviceTime().count();
  std::cout << "Copy to Device Time: " << copy_to_device_time_s * 1000 << " ms"
            << std::endl;

  double device_compute_time_s = GetDeviceComputeTime().count();
  std::cout << "Device Compute Time: " << device_compute_time_s * 1000 << " ms"
            << std::endl;

  double copy_from_device_time_s = GetCopyFromDeviceTime().count();
  std::cout << "Copy From Devcie Time: " << copy_from_device_time_s * 1000
            << " ms" << std::endl;
}

const ProgramArgs &CudaPrefixSumSolver::program_args() const {
  return program_args_;
}
