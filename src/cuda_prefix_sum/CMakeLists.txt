# CUDA prefix sum library target

add_library(prefix_sum_cuda
    kernel_launch_params.cpp
    cuda_prefix_sum_solver.cu
    cuda_prefix_sum_solver_single_element.cu
    cuda_prefix_sum_solver.cpp
    cuda_prefix_sum_warp_solver.cpp
    cuda_prefix_sum_warp_solver.cu
)

# Properly attach include paths
target_include_directories(prefix_sum_cuda
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

# Link against CUDA runtime + any other libs
target_link_libraries(prefix_sum_cuda
    PUBLIC
        CLI11::CLI11
        CUDA::cudart                        # CUDA runtime from CUDAToolkit
        cuda_headers
    PRIVATE
        my_default_flags
)


