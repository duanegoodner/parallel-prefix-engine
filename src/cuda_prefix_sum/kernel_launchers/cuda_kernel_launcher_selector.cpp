

#include "cuda_prefix_sum/cuda_kernel_launcher_selector.hpp"

#include "cuda_prefix_sum/single_tile_kernel_launcher.cuh"
#include "cuda_prefix_sum/multi_tile_kernel_launcher.cuh"

#include <stdexcept>
#include <string>
#include <unordered_map>

std::unique_ptr<KernelLauncher> CreateCudaKernelLauncher(const ProgramArgs& args) {
    using LauncherCreator = std::function<std::unique_ptr<KernelLauncher>()>;

    static const std::unordered_map<std::string, LauncherCreator> kernel_map = {
        { "single_tile", [] { return std::make_unique<SingleTileKernelLauncher>(); } },
        { "multi_tile",  [] { return std::make_unique<MultiTileKernelLauncher>(); } }
        // Add more kernel launchers here as needed
    };

    const auto& kernel = args.cuda_kernel().value_or("");
    auto it = kernel_map.find(kernel);
    if (it == kernel_map.end()) {
        throw std::runtime_error("Unsupported CUDA kernel: " + kernel);
    }

    return it->second();
}
