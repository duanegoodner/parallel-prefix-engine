#include "cuda_prefix_sum/cuda_prefix_sum_solver.hpp"

#include <chrono>
#include <iostream>

// Constructor: store parsed CLI args
CudaPrefixSumSolver::CudaPrefixSumSolver(int argc, char* argv[]) {
  args_ = ProgramArgs::Parse(argc, argv);
}

void CudaPrefixSumSolver::StartTimer() {
  // We'll fill this in once we launch a kernel
}

void CudaPrefixSumSolver::Compute(std::vector<int>& local_matrix) {
  // TODO: Launch CUDA kernel here in next steps
  // For now, just log placeholder
  std::cerr << "[CUDA] Compute() not yet implemented.\n";
}

void CudaPrefixSumSolver::StopTimer() {
  // Placeholder - real logic will come after kernel
}

void CudaPrefixSumSolver::ReportTime() const {
  std::cout << "CUDA Execution time (placeholder): " << execution_time_ms_
            << " ms\n";
}

void CudaPrefixSumSolver::PrintMatrix(const std::vector<int>& local_matrix,
                                      const std::string& header) const {
  std::cout << header << "\n";
  int local_n = args_.local_n();
  for (int row = 0; row < local_n; ++row) {
    std::cout << "\t";
    for (int col = 0; col < local_n; ++col) {
      std::cout << local_matrix[row * local_n + col] << "\t";
    }
    std::cout << "\n";
  }
}
