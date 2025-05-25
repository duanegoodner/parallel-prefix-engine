add_executable(test_sub_tile_kernels test_sub_tile_kernels.cu)

target_include_directories(
    test_sub_tile_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/cuda_prefix_sum/include
)

target_link_libraries(
    test_sub_tile_kernels
    PRIVATE
    GTest::gtest_main
    CUDA::cudart
    cuda_headers
    kernel_launch_params
    my_default_flags
    sub_tile_kernels
)