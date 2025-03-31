#include "common/program_args.hpp"
#include "mpi_prefix_sum/mpi_utils.hpp"
#include <iostream>
#include <string>
#include <cstdlib>

ProgramArgs::ProgramArgs(int local_n, int seed, std::string backend)
    : local_n_(local_n), seed_(seed), backend_(std::move(backend)) {}

void ProgramArgs::PrintUsage(std::ostream& os) {
  os << "Usage:\n"
     << "  ./prefix_sum_mpi <local_n> [seed] [--backend=mpi|cuda]\n\n"
     << "Arguments:\n"
     << "  <local_n>   Size of each local submatrix (e.g., 2, 4)\n"
     << "  [seed]      Optional seed for random matrix init\n"
     << "  [--backend=...]  Optional backend selection (default: mpi)\n";
}

ProgramArgs ProgramArgs::Parse(int argc, char* const argv[]) {
  if (argc < 2) {
    PrintUsage(std::cerr);
    std::exit(1);
  }

  int local_n = std::stoi(argv[1]);
  int seed = (argc >= 3 && argv[2][0] != '-') ? std::stoi(argv[2]) : 1234;

  std::string backend = "mpi";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--backend=", 0) == 0) {
      backend = arg.substr(10);
    }
  }

  return ProgramArgs(local_n, seed, backend);
}

ProgramArgs ProgramArgs::ParseForMPI(int argc, char* const argv[], int rank) {
  if (argc < 2) {
    if (rank == 0) PrintUsage(std::cerr);
    std::exit(1);
  }
  return Parse(argc, argv);
}
