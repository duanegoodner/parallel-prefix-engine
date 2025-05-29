#pragma once

#include "common/prefix_sum_solver.hpp"
#include "common/program_args.hpp"
#include "common/time_utils.hpp"

class ZScaffoldPrefixSumSolver : public PrefixSumSolver {
  ZScaffoldPrefixSumSolver(
      const ProgramArgs &program_args
  ); // constructor can take additional args if needed

  void PopulateFullMatrix() override;
  void Compute() override;
  void PrintFullMatrix(std::string title = "") override;
  void PrintLowerRightElement(std::string title = "") override;
  void StartTimer() override;
  void StopTimer() override;
  void ReportTime() const override;
};