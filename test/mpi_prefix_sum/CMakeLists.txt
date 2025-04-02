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
    prefix_sum_common
    prefix_sum_mpi
    )