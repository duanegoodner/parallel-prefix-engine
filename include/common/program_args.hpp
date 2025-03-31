#pragma once

#include "common/prefix_sum_solver.hpp"

#include <iostream>
#include <memory>
#include <string>

class ProgramArgs {
public:
  ProgramArgs() = default;
  ProgramArgs(int local_n, int seed, std::string backend, bool verbose);

  static ProgramArgs Parse(int argc, char *const argv[]);

  [[nodiscard]] int local_n() const { return local_n_; }
  [[nodiscard]] int seed() const { return seed_; }
  [[nodiscard]] const std::string &backend() const { return backend_; }
  bool verbose() const { return verbose_; }

  std::unique_ptr<PrefixSumSolver> MakeSolver(int argc, char *argv[]) const;

private:
  int local_n_ = 0;
  int seed_ = 1234;
  std::string backend_ = "mpi";
  bool verbose_ = false;
};
