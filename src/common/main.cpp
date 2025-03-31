#include "common/matrix_init.hpp"
#include "common/prefix_sum_solver.hpp"
#include "common/program_args.hpp"

#include <iostream>
#include <memory>
#include <vector>

int main(int argc, char *argv[]) {
  ProgramArgs args = ProgramArgs::Parse(argc, argv);

  if (args.verbose()) {
    std::cout << "Parsed options:\n"
              << "  local_n : " << args.local_n() << "\n"
              << "  seed    : " << args.seed() << "\n"
              << "  backend : " << args.backend() << "\n";
  }

  auto solver = args.MakeSolver(argc, argv);
  auto local_mat = GenerateRandomMatrix<int>(args.local_n(), args.seed());

  solver->PrintMatrix(local_mat, "Before prefix sum:");
  solver->Compute(local_mat);
  solver->PrintMatrix(local_mat, "After prefix sum:");

  return 0;
}
