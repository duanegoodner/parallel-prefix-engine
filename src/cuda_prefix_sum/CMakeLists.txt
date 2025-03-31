# CUDA prefix sum library target

add_library(prefix_sum_cuda
    cuda_prefix_sum_solver.cu
    # cuda_utils.cpp
)

target_include_directories(prefix_sum_cuda
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(prefix_sum_cuda
    PUBLIC
        CLI11::CLI11
)
