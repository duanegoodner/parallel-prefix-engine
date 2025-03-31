#pragma once

#include <string>
#include <iostream>

class ProgramArgs {
 public:
  ProgramArgs() = default;
  ProgramArgs(int local_n, int seed, std::string backend);

  static ProgramArgs Parse(int argc, char* const argv[]);
  static void PrintUsage(std::ostream& os);

  [[nodiscard]] int local_n() const { return local_n_; }
  [[nodiscard]] int seed() const { return seed_; }
  [[nodiscard]] const std::string& backend() const { return backend_; }

 private:
  int local_n_ = 0;
  int seed_ = 1234;
  std::string backend_ = "mpi";
};
