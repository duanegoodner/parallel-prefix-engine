add_executable(test_matrix_init
    test_matrix_init.cpp
)

target_include_directories(test_matrix_init
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(test_matrix_init
    PRIVATE
    GTest::gtest_main
    my_default_flags
)

# Additional link dependencies
gtest_discover_tests(test_matrix_init)
