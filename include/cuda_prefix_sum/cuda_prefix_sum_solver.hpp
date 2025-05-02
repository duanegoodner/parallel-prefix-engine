// cuda_prefix_sum_solver.hpp
//
// Declares the CudaPrefixSumSolver class for performing distributed prefix
// sum using CUDA. Implements the PrefixSumSolver interface.

#pragma once

#include <chrono>
#include <string>
#include <vector>

#include "common/prefix_sum_solver.hpp"
#include "common/program_args.hpp"

class CudaPrefixSumSolver : public PrefixSumSolver {
public:
  explicit CudaPrefixSumSolver(const ProgramArgs &program_args);

  void PopulateFullMatrix() override;
  void Compute() override;

  void PrintFullMatrix(std::string title = "") override;

  const ProgramArgs &program_args() const;

  void StartTimer() override;
  void StopTimer() override;
  void StartCopyToDeviceTimer();
  void StopCopyToDeviceTimer();
  void StartDeviceComputeTimer();
  void StopDeviceComputeTimer();
  void StartCopyFromDeviceTimer();
  void StopCopyFromDeviceTimer();

  std::chrono::duration<double> GetElapsedTime() const;
  std::chrono::duration<double> GetStartTime() const override;
  std::chrono::duration<double> GetEndTime() const override;

  std::chrono::duration<double> GetCopyToDeviceTime() const;
  std::chrono::duration<double> GetDeviceComputeTime() const;
  std::chrono::duration<double> GetCopyFromDeviceTime() const;

  void ReportTime() const override;

private:
  ProgramArgs program_args_;
  std::vector<int> full_matrix_;
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point end_time_;

  std::chrono::steady_clock::time_point copy_to_device_start_time_,
      copy_to_device_end_time_;
  std::chrono::steady_clock::time_point device_compute_start_time_, device_compute_end_time_;
  std::chrono::steady_clock::time_point copy_from_device_start_time_,
      copy_from_device_end_time_;
};
