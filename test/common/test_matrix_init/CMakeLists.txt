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
    program_args
    )
target_link_libraries(test_matrix_init PRIVATE my_default_flags)

gtest_discover_tests(test_matrix_init)
