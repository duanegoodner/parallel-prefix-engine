add_executable(test_subtile_kernel test_subtile_kernel.cpp)

target_include_directories(
    test_subtile_kernel PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/cuda_prefix_sum/include
    )

target_link_libraries(
    test_subtile_kernel PRIVATE
    GTest::gtest_main
)