#pragma once

#include <string>

class ProgramArgs {
public:
  ProgramArgs() = default;
  ProgramArgs(int local_n, int seed, std::string backend);

  static ProgramArgs Parse(int argc, char *const argv[]);
  static ProgramArgs ParseForMPI(int argc, char *const argv[], int rank);

  [[nodiscard]] int local_n() const { return local_n_; }
  [[nodiscard]] int seed() const { return seed_; }
  [[nodiscard]] const std::string &backend() const { return backend_; }

  static void PrintUsage(std::ostream &os);

private:
  int local_n_ = 0;
  int seed_ = 1234;
  std::string backend_ = "mpi";
};