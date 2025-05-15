// cuda_prefix_sum_solver.cuh
#pragma once

#include <cuda_runtime.h>

#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"

void LaunchSingleTileKernel(KernelLaunchParams kernel_params);

void LaunchPrefixSumKernelSingleElement(KernelLaunchParams kernel_params);

// void LaunchPrefixSumKernelWarp(KernelLaunchParams kernel_params);

// void LaunchPrefixSumKernelWarpNaive(KernelLaunchParams kernel_params);

void LaunchPrefixSumKernelAccum(KernelLaunchParams kernel_params);

void LaunchPrefixSumKernelHierarchical(KernelLaunchParams params);

void LaunchPrefixSumKernelHybrid(KernelLaunchParams params);

