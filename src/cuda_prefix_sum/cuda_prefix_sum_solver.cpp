#include "cuda_prefix_sum/cuda_prefix_sum_solver.hpp"

#include <cuda_runtime.h> // Required for cudaStream_t

#include <iostream>
#include <memory>

#include "common/matrix_init.hpp"

// #include "cuda_prefix_sum/cuda_prefix_sum_solver.cuh"
#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"
#include "cuda_prefix_sum/kernel_launcher.hpp"

// Ensure proper linkage between C++ and CUDA code
// void LaunchPrefixSumKernel(int* d_data, int tile_dim, cudaStream_t stream =
// 0);

CudaPrefixSumSolver::CudaPrefixSumSolver(
    const ProgramArgs &program_args,
    std::unique_ptr<KernelLauncher> kernel_launcher
    // KernelLaunchFunction kernel_launch_func
)
    : program_args_(program_args)
    , kernel_launcher_(std::move(kernel_launcher)) {
  PopulateFullMatrix();
  AllocateDeviceMemory();
  std::vector<std::string> time_interval_names{
      "warmup",
      "total",
      "copy_to_device",
      "compute",
      "copy_from_device"};
  time_intervals_.AttachIntervals(time_interval_names);
  WarmUp();
}

CudaPrefixSumSolver::~CudaPrefixSumSolver() { FreeDeviceMemory(); }

void CudaPrefixSumSolver::AllocateDeviceMemory() {
  cudaMalloc(&device_data_, program_args_.FullMatrixSize1D() * sizeof(int));
}

void CudaPrefixSumSolver::FreeDeviceMemory() {
  if (device_data_) {
    cudaFree(device_data_);
    device_data_ = nullptr;
  }
}

void CudaPrefixSumSolver::PopulateFullMatrix() {
  full_matrix_ = GenerateRandomMatrix<int>(
      program_args_.full_matrix_dim()[0],
      program_args_.full_matrix_dim()[1],
      program_args_.seed()
  );
}

void CudaPrefixSumSolver::WarmUp() {
  time_intervals_.RecordStart("warmup");
  cudaFree(0);
  time_intervals_.RecordEnd("warmup");
}

void CudaPrefixSumSolver::CopyDataFromHostToDevice() {
  time_intervals_.RecordStart("copy_to_device");
  cudaMemcpy(
      device_data_,
      full_matrix_.data(),
      program_args_.FullMatrixSize1D() * sizeof(int),
      cudaMemcpyHostToDevice
  );
  time_intervals_.RecordEnd("copy_to_device");
}

void CudaPrefixSumSolver::RunKernel() {
  time_intervals_.RecordStart("compute");
  // auto launch_params = CreateKernelLaunchParams(device_data_, program_args_);
  // kernel_launch_func_(launch_params);
  kernel_launcher_->Launch(device_data_);
  cudaDeviceSynchronize();
  time_intervals_.RecordEnd("compute");
}

void CudaPrefixSumSolver::CopyDataFromDeviceToHost() {
  time_intervals_.RecordStart("copy_from_device");
  cudaMemcpy(
      full_matrix_.data(),
      device_data_,
      program_args_.FullMatrixSize1D() * sizeof(int),
      cudaMemcpyDeviceToHost
  );
  time_intervals_.RecordEnd("copy_from_device");
}

void CudaPrefixSumSolver::Compute() {
  CopyDataFromHostToDevice();
  RunKernel();
  CopyDataFromDeviceToHost();
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
  time_intervals_.RecordStart("total");
}

void CudaPrefixSumSolver::StopTimer() { time_intervals_.RecordEnd("total"); }

void CudaPrefixSumSolver::ReportTime() const {
  double elapsed_time_s = time_intervals_.ElapsedTime("total").count();
  std::cout << "\n=== Runtime Report ===\n";
  std::cout << "Total Time: " << elapsed_time_s * 1000 << " ms" << std::endl;

  double copy_to_device_time_s =
      time_intervals_.ElapsedTime("copy_to_device").count();
  std::cout << "Copy to Device Time: " << copy_to_device_time_s * 1000 << " ms"
            << std::endl;

  double device_compute_time_s =
      time_intervals_.ElapsedTime("compute").count();
  std::cout << "Device Compute Time: " << device_compute_time_s * 1000 << " ms"
            << std::endl;

  double copy_from_device_time_s =
      time_intervals_.ElapsedTime("copy_from_device").count();
  std::cout << "Copy From Devcie Time: " << copy_from_device_time_s * 1000
            << " ms" << std::endl;
}

const ProgramArgs &CudaPrefixSumSolver::program_args() const {
  return program_args_;
}
