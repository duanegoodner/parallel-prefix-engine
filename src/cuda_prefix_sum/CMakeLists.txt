# CUDA prefix sum library target

add_library(prefix_sum_cuda
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


add_executable(
    cuda_functionality
    cuda_prefix_sum_solver.cu
    cuda_prefix_sum_solver.cpp
    cuda_functionality.cpp)

target_include_directories(cuda_functionality
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )

target_link_libraries(
    cuda_functionality
    PRIVATE
    prefix_sum_common
    prefix_sum_mpi
    prefix_sum_cuda
)
