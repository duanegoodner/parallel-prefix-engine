// ----------------------------------------------------------------------------
// program_args.hpp
//
// Program args definitions.
// This header is part of the prefix sum project.
// ----------------------------------------------------------------------------

#pragma once

#include <iostream>
#include <memory>
#include <string>

#include "common/prefix_sum_solver.hpp"

// Class ProgramArgs: Parses and stores command-line arguments such as local
// matrix size, seed, and backend.
class ProgramArgs {
public:
  ProgramArgs() = default;
  ProgramArgs(
      int local_n,
      int seed,
      std::string backend,
      bool verbose,
      int orig_argc,
      char **orig_argv
  );

  static ProgramArgs Parse(int argc, char *const argv[]);

  [[nodiscard]] int local_n() const { return local_n_; }
  [[nodiscard]] int seed() const { return seed_; }
  [[nodiscard]] const std::string &backend() const { return backend_; }
  bool verbose() const { return verbose_; }
  [[nodiscard]] int orig_argc() const { return orig_argc_; }
  [[nodiscard]] char **orig_argv() const { return orig_argv_; }

  std::unique_ptr<PrefixSumSolver> MakeSolver() const;

private:
  int local_n_ = 0;
  int seed_ = 1234;
  std::string backend_ = "mpi";
  bool verbose_ = false;
  int orig_argc_ = 0;
  char **orig_argv_ = nullptr;
};