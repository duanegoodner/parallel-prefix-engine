# CUDA prefix sum library target

add_subdirectory(kernel_array)
add_subdirectory(kernel_launch_params)
add_subdirectory(kernel_launchers)
add_subdirectory(kernels)
add_subdirectory(registration)

add_library(prefix_sum_cuda
    cuda_prefix_sum_solver.cpp
)

# Properly attach include paths
target_include_directories(prefix_sum_cuda
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link against CUDA runtime + any other libs
target_link_libraries(prefix_sum_cuda
    PUBLIC
        CLI11::CLI11
    PRIVATE
        cuda_solver_registration
        kernel_launch_params
        my_default_flags
        CUDA::cudart                       
        cuda_headers
)


