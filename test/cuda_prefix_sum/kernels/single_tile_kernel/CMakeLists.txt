add_executable(test_single_tile_kernel test_single_tile_kernel.cpp)

target_include_directories(
    test_single_tile_kernel PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/cuda_prefix_sum/include
    )

target_link_libraries(
    test_single_tile_kernel PRIVATE
    GTest::gtest_main
)