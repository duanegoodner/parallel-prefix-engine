#pragma once
#include <memory>

#include "common/prefix_sum_solver.hpp"
#include "common/program_args.hpp"

std::unique_ptr<PrefixSumSolver> MakeSolver(ProgramArgs &program_args);