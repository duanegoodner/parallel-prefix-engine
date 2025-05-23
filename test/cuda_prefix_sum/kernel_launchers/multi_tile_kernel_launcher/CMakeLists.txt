add_executable(test_multi_tile_kernel_launcher test_multi_tile_kernel_launcher.cpp)

target_include_directories(
    test_multi_tile_kernel_launcher PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/cuda_prefix_sum/include
)

target_link_libraries(
    test_multi_tile_kernel_launcher
    PRIVATE
    GTest::gtest_main
    CUDA::cudart
    cuda_headers
    my_default_flags
    sub_tile_kernels
    row_scan_single_block_kernel
    row_scan_multi_block_kernel
    multi_tile_kernel_launcher
)