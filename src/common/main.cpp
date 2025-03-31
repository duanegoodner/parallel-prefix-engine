#include "common/matrix_init.hpp"
#include "common/prefix_sum_solver.hpp"
#include "mpi_prefix_sum/mpi_prefix_sum_solver.hpp"

#include <memory>
#include <vector>

int main(int argc, char* argv[]) {
  // Create the solver (initializes MPI and parses args)
  std::unique_ptr<PrefixSumSolver> solver =
      std::make_unique<MpiPrefixSumSolver>(argc, argv);

  // Access parsed arguments (needed to size and seed matrix)
  const ProgramArgs& args = static_cast<MpiPrefixSumSolver*>(solver.get())->args();

  // Generate local matrix filled with random integers
  auto local_mat = GenerateRandomMatrix<int>(args.local_n(), args.seed());

  // Print before prefix sum
  solver->PrintMatrix(local_mat, "Before prefix sum:");

  // Run prefix sum algorithm
  solver->Compute(local_mat);

  // Print after prefix sum
  solver->PrintMatrix(local_mat, "After prefix sum:");

  return 0;
}
