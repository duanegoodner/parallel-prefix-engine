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

  void PopulateFullMatrix() override;
  void Compute() override;

  void PrintFullMatrix(std::string title = "") override;

  const ProgramArgs &program_args() const;

  void WarmUp() override;
  void StartTimer() override;
  void StopTimer() override;

  void ReportTime() const override;

private:
  ProgramArgs program_args_;
  std::vector<int> full_matrix_;
  std::unordered_map<std::string, TimeInterval> time_intervals_;
  KernelLaunchFunction kernel_launch_func_;
  void AttachTimeInterval(std::string name);

  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point end_time_;

  std::chrono::steady_clock::time_point copy_to_device_start_time_,
      copy_to_device_end_time_;
  std::chrono::steady_clock::time_point device_compute_start_time_,
      device_compute_end_time_;
  std::chrono::steady_clock::time_point copy_from_device_start_time_,
      copy_from_device_end_time_;
};
