#pragma once

#include <memory>
#include "cuda_prefix_sum/kernel_launcher.hpp"
#include "common/program_args.hpp"

std::unique_ptr<KernelLauncher> CreateCudaKernelLauncher(const ProgramArgs&);
