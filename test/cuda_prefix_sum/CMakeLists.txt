add_subdirectory(kernels)
add_subdirectory(kernel_launchers)

# ===========================
# CUDA_PREFIX_SUM_SOLVER_SINGLE_TILE
# ===========================

add_executable(
    test_cuda_prefix_sum_solver_single_tile
    test_cuda_prefix_sum_solver_single_tile.cpp
    )
target_include_directories(
    test_cuda_prefix_sum_solver_single_tile PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/cuda_prefix_sum/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )
target_link_libraries(
    test_cuda_prefix_sum_solver_single_tile
    PRIVATE
    GTest::gtest_main
    program_args
    prefix_sum_mpi
    prefix_sum_cuda
    single_tile_kernel
    single_tile_kernel_launcher
    time_utils
    my_default_flags
    CUDA::cudart                       
    cuda_headers
    )

gtest_discover_tests(test_cuda_prefix_sum_solver_single_tile)


# ===========================
# CUDA_PREFIX_SUM_SOLVER_MULTI_TILE
# ===========================

add_executable(
    test_cuda_prefix_sum_solver_multi_tile
    test_cuda_prefix_sum_solver_multi_tile.cpp
    )
target_include_directories(
    test_cuda_prefix_sum_solver_multi_tile PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/cuda_prefix_sum/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )
target_link_libraries(
    test_cuda_prefix_sum_solver_multi_tile
    PRIVATE
    GTest::gtest_main
    program_args
    prefix_sum_mpi
    prefix_sum_cuda
    multi_tile_kernel
    multi_tile_kernel_launcher
    time_utils
    my_default_flags
    CUDA::cudart                       
    cuda_headers
    )

gtest_discover_tests(test_cuda_prefix_sum_solver_multi_tile)

