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
    prefix_sum_common
    prefix_sum_mpi
    prefix_sum_cuda
    )
target_link_libraries(test_cuda_prefix_sum_solver PRIVATE my_default_flags)


# ===========================
# CUDA_PREFIX_SUM_WARP_SOLVER
# ===========================

add_executable(
    test_cuda_prefix_sum_warp_solver
    test_cuda_prefix_sum_warp_solver.cpp
    )
target_include_directories(
    test_cuda_prefix_sum_warp_solver PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )
target_link_libraries(
    test_cuda_prefix_sum_warp_solver
    PRIVATE
    GTest::gtest_main
    prefix_sum_common
    prefix_sum_mpi
    prefix_sum_cuda
    )
target_link_libraries(test_cuda_prefix_sum_warp_solver PRIVATE my_default_flags)