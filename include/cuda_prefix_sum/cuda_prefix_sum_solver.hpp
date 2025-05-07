// cuda_prefix_sum_solver.hpp
//
// Declares the CudaPrefixSumSolver class for performing distributed prefix
// sum using CUDA. Implements the PrefixSumSolver interface.

#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/prefix_sum_solver.hpp"
#include "common/program_args.hpp"
#include "common/time_utils.hpp"

#include "cuda_prefix_sum/kernel_launch_params.hpp"

class CudaPrefixSumSolver : public PrefixSumSolver {
public:
  using KernelLaunchFunction =
      void (*)(const KernelLaunchParams);

  explicit CudaPrefixSumSolver(
      const ProgramArgs &program_args,
      KernelLaunchFunction kernel_launch_func
  );

  ~CudaPrefixSumSolver();

  void PrintFullMatrix(std::string title = "") override;
  void PopulateFullMatrix() override;
  void StartTimer() override;
  void Compute() override;
  void StopTimer() override;
  void ReportTime() const override;

  const ProgramArgs &program_args() const;
  void WarmUp();

private:
  ProgramArgs program_args_;
  std::vector<int> full_matrix_;
  TimeIntervals time_intervals_;
  KernelLaunchFunction kernel_launch_func_;

  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point end_time_;
  int *device_data_ = nullptr;

  void AllocateDeviceMemory();
  void CopyDataFromHostToDevice();
  void CopyDataFromDeviceToHost();
  void RunKernel();
  void FreeDeviceMemory();

};
