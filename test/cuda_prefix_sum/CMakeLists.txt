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
    )
target_link_libraries(
    test_cuda_prefix_sum_solver
    PRIVATE
    GTest::gtest_main
    prefix_sum_common
    prefix_sum_cuda
    )