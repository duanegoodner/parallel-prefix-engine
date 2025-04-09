# CUDA prefix sum library target

add_library(prefix_sum_cuda
    kernel_launch_params.cpp
    cuda_prefix_sum_solver.cu
    cuda_prefix_sum_solver.cpp
)

target_include_directories(prefix_sum_cuda
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(prefix_sum_cuda
    PUBLIC
        CLI11::CLI11
)


