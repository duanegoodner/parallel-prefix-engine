#include "cuda_prefix_sum/subtile_kernel.cuh"
#include "cuda_prefix_sum/subtile_kernel_launcher.cuh"
#include "cuda_prefix_sum/kernel_config_utils.cuh"
#include "cuda_prefix_sum/internal/kernel_launch_params.hpp"

void SubTileKernelLauncher::Launch(const KernelLaunchParams &launch_params) {
    // Set max dynamic shared memory and prefer shared over L1
    constexpr size_t kMaxSharedMemBytes = 98304;
    ConfigureSharedMemoryForKernel(SubtileKernel, kMaxSharedMemBytes);

    // Prepare launch configuration
    dim3 block_dim = GetBlockDim(launch_params);
    dim3 grid_dim = GetGridDim(launch_params);
    size_t shared_mem_size = GetSharedMemSize(launch_params);

    // Launch the kernel
    SubtileKernel<<<grid_dim, block_dim, shared_mem_size, 0>>>(launch_params);

    // Validate
    CheckErrors();
}
