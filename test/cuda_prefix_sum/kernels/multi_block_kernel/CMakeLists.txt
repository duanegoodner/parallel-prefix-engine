add_executable(test_multi_block_kernel test_multi_block_kernel.cpp)

target_include_directories(
    test_multi_block_kernel PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/cuda_prefix_sum/include
    )

target_link_libraries(
    test_multi_block_kernel
    PRIVATE
    GTest::gtest_main
    multi_tile_kernel
    my_default_flags
    CUDA::cudart
    cuda_headers
)