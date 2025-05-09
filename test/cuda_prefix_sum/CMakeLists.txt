# ===========================
# CUDA_PREFIX_SUM_SOLVER
# ===========================

add_executable(
    test_cuda_prefix_sum_solver
    test_cuda_prefix_sum_solver.cpp
    )
target_include_directories(
    test_cuda_prefix_sum_solver PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )
target_link_libraries(
    test_cuda_prefix_sum_solver
    PRIVATE
    GTest::gtest_main
    program_args
    prefix_sum_mpi
    prefix_sum_cuda
    subtile_kernel
    subtile_kernel_launcher
    time_utils
    my_default_flags
    CUDA::cudart                       
    cuda_headers
    )

