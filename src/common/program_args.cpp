#include "common/program_args.hpp"

#include <CLI/CLI.hpp>
#include <utility>  // for std::move

ProgramArgs::ProgramArgs(int local_n, int seed, std::string backend)
    : local_n_(local_n), seed_(seed), backend_(std::move(backend)) {}

void ProgramArgs::PrintUsage(std::ostream& os) {
  os << "Usage:\n"
     << "  ./prefix_sum_mpi <local_n> [seed] [--backend=mpi|cuda]\n\n"
     << "Arguments:\n"
     << "  <local_n>   Size of each local submatrix (NxN per rank)\n"
     << "  [seed]      Optional seed for random generation\n"
     << "  [--backend] Backend to use (default: mpi)\n";
}

ProgramArgs ProgramArgs::Parse(int argc, char* const argv[]) {
  CLI::App app{"Distributed prefix sum runner"};

  int local_n;
  int seed = 1234;
  std::string backend = "mpi";

  app.add_option("local_n", local_n, "Size of local matrix")->required();
  app.add_option("seed", seed, "Optional seed");
  app.add_option("--backend", backend, "Backend to use")
     ->check(CLI::IsMember({"mpi"}))
     ->default_val("mpi");

     try {
      app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
      std::exit(app.exit(e));  // will print error and help message
    }
    

  return ProgramArgs(local_n, seed, backend);
}
