#pragma once
#include <cuda_runtime.h>

#include <cstdio>

template <typename KernelFunc>
inline void ConfigureSharedMemoryForKernel(
    KernelFunc kernel_func,
    int shared_mem_bytes
) {
  cudaError_t err;

  err = cudaFuncSetCacheConfig(kernel_func, cudaFuncCachePreferShared);
  if (err != cudaSuccess) {
    fprintf(
        stderr,
        "Failed to set cache config: %s\n",
        cudaGetErrorString(err)
    );
  }

  err = cudaFuncSetAttribute(
      kernel_func,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_mem_bytes
  );
  if (err != cudaSuccess) {
    fprintf(
        stderr,
        "Failed to set max shared memory: %s\n",
        cudaGetErrorString(err)
    );
  }
}
