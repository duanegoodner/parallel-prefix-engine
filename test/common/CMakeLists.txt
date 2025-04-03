# ===========================
# MATRIX_INIT
# ===========================

add_executable(
    test_matrix_init
    test_matrix_init.cpp
    )
target_include_directories(
    test_matrix_init PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    )
target_link_libraries(
    test_matrix_init
    PRIVATE
    GTest::gtest_main
    prefix_sum_common
    )

# ===========================
# MATRIX_OUTPUT
# ===========================

add_executable(
    test_matrix_output
    test_matrix_output.cpp
    )
target_include_directories(
    test_matrix_output PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    )
target_link_libraries(
    test_matrix_output
    PRIVATE
    GTest::gtest_main
    prefix_sum_common
    )


# ===========================
# PROGRAM_ARGS
# ===========================

add_executable(
    test_program_args
    test_program_args.cpp
    )
target_include_directories(
    test_program_args PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    )
target_link_libraries(
    test_program_args
    PRIVATE
    GTest::gtest_main
    prefix_sum_common
    prefix_sum_mpi
    prefix_sum_cuda
    )