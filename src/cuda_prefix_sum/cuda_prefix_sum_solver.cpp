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
  AttachTimeInterval("warmup");
  AttachTimeInterval("total");
  AttachTimeInterval("copy_to_device");
  AttachTimeInterval("compute");
  AttachTimeInterval("copy_from_device");
}

void CudaPrefixSumSolver::PopulateFullMatrix() {
  full_matrix_ = GenerateRandomMatrix<int>(
      program_args_.full_matrix_dim()[0],
      program_args_.full_matrix_dim()[1],
      program_args_.seed()
  );
}

void CudaPrefixSumSolver::AttachTimeInterval(std::string name) {
  time_intervals_[name] = TimeInterval();
}

void CudaPrefixSumSolver::Compute() {
  int *d_data = nullptr;

  // StartCopyToDeviceTimer();
  time_intervals_.at("copy_to_device").RecordStart();

  // Allocate device memory
  cudaMalloc(&d_data, program_args_.FullMatrixSize() * sizeof(int));
  cudaMemcpy(
      d_data,
      full_matrix_.data(),
      program_args_.FullMatrixSize() * sizeof(int),
      cudaMemcpyHostToDevice
  );

  time_intervals_.at("copy_to_device").RecordEnd();
  // StopCopyToDeviceTimer();

  // StartDeviceComputeTimer();
  time_intervals_.at("compute").RecordStart();

  auto launch_params = CreateKernelLaunchParams(d_data, program_args_);

  // Launch kernel
  LaunchPrefixSumKernel(launch_params, 0);
  // LaunchPrefixSumKernelSingleElement(launch_params, 0);

  cudaDeviceSynchronize();

  // StopDeviceComputeTimer();
  time_intervals_.at("compute").RecordEnd();

  // StartCopyFromDeviceTimer();
  time_intervals_.at("copy_from_device").RecordStart();

  // Copy results back
  cudaMemcpy(
      full_matrix_.data(),
      d_data,
      program_args_.FullMatrixSize() * sizeof(int),
      cudaMemcpyDeviceToHost
  );

  cudaFree(d_data);

  // StopCopyFromDeviceTimer();
  time_intervals_.at("copy_from_device").RecordEnd();
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

void CudaPrefixSumSolver::WarmUp() {
  time_intervals_.at("warmup").RecordStart();
  cudaFree(0);
  time_intervals_.at("warmup").RecordEnd();
}

void CudaPrefixSumSolver::StartTimer() {
  time_intervals_["total"].RecordStart();
}

void CudaPrefixSumSolver::StopTimer() { time_intervals_["total"].RecordEnd(); }

void CudaPrefixSumSolver::ReportTime() const {
  double elapsed_time_s = time_intervals_.at("total").ElapsedTime().count();
  std::cout << "\n=== Runtime Report ===\n";
  std::cout << "Total Time: " << elapsed_time_s * 1000 << " ms" << std::endl;

  double copy_to_device_time_s =
      time_intervals_.at("copy_to_device").ElapsedTime().count();
  std::cout << "Copy to Device Time: " << copy_to_device_time_s * 1000 << " ms"
            << std::endl;

  double device_compute_time_s =
      time_intervals_.at("compute").ElapsedTime().count();
  std::cout << "Device Compute Time: " << device_compute_time_s * 1000 << " ms"
            << std::endl;

  double copy_from_device_time_s =
      time_intervals_.at("copy_from_device").ElapsedTime().count();
  std::cout << "Copy From Devcie Time: " << copy_from_device_time_s * 1000
            << " ms" << std::endl;
}

const ProgramArgs &CudaPrefixSumSolver::program_args() const {
  return program_args_;
}
