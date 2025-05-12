# ===========================
# PREFIX_SUM_BLOCK_MATRIX
# ===========================

add_executable(
    test_prefix_sum_block_matrix
    test_prefix_sum_block_matrix.cpp
    )
target_include_directories(
    test_prefix_sum_block_matrix PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    )
target_link_libraries(
    test_prefix_sum_block_matrix
    PRIVATE
    GTest::gtest_main
    program_args
    prefix_sum_mpi
    MPI::MPI_CXX
    )
target_link_libraries(test_prefix_sum_block_matrix PRIVATE my_default_flags)

gtest_discover_tests(test_prefix_sum_block_matrix)