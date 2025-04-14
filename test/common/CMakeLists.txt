
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
target_link_libraries(test_matrix_init PRIVATE my_default_flags)


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
    ${CMAKE_CURRENT_SOURCE_DIR}/../include

    )
target_link_libraries(
    test_program_args
    PRIVATE
    GTest::gtest_main
    prefix_sum_common
    prefix_sum_mpi
    prefix_sum_cuda
    )
target_link_libraries(test_program_args PRIVATE my_default_flags)


gtest_discover_tests(test_matrix_init)
gtest_discover_tests(test_program_args)

