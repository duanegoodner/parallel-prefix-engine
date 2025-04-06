#include <iostream>

#include "common/program_args.hpp"
#include "cuda_prefix_sum/cuda_prefix_sum_solver.hpp"


int main() {
    std::cout << "Preparing to check CUDA functionality" << std::endl;

    ProgramArgs program_args{};
    std::cout << "ProgramArgs instantiated" << std::endl;
    

    CudaPrefixSumSolver cuda_solver{program_args};
    std::cout << "CudaPrefixSumSolver instantiated" << std::endl;

    std::cout << "Initial full matrix in host memory:" << std::endl;
    cuda_solver.PrintFullMatrix();

    std::cout << "Calling CudaPrefixSumSolver.ComputeNew()" << std::endl;
    cuda_solver.ComputeNew();

    std::cout << "Done with current checks" << std::endl;

    

    return 0;
}