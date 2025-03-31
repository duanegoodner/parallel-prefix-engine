#include "common/logger.hpp"
#include "common/matrix_init.hpp"
#include "common/prefix_sum_solver.hpp"
#include "common/program_args.hpp"

#include <iostream>
#include <memory>
#include <vector>

int main(int argc, char *argv[]) {
  ProgramArgs args = ProgramArgs::Parse(argc, argv);
  // Logger::SetVerbose(args.verbose()); // 👈 enable debug messages if
  // requested

  // Logger::Log(
  //     LogLevel::INFO,
  //     "Creating solver for backend: " + args.backend()
  // );

  if (args.verbose()) {
    std::cout << "Parsed options:\n"
              << "  local_n : " << args.local_n() << "\n"
              << "  seed    : " << args.seed() << "\n"
              << "  backend : " << args.backend() << "\n"
              << std::endl;
  }

  auto solver = args.MakeSolver(argc, argv);
  auto local_mat = GenerateRandomMatrix<int>(args.local_n(), args.seed());

  // Logger::Log(LogLevel::DEBUG, "Random matrix initialized.");

  solver->PrintMatrix(local_mat, "Before prefix sum:");
  solver->StartTimer();
  solver->Compute(local_mat);
  solver->StopTimer();
  solver->ReportTime();
  solver->PrintMatrix(local_mat, "After prefix sum:");

  return 0;
}
