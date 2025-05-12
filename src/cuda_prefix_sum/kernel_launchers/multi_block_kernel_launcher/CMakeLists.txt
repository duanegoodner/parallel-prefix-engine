add_library(multi_block_kernel_launcher multi_block_kernel_launcher.cu)

target_include_directories(
    multi_block_kernel_launcher
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(multi_block_kernel_launcher
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)


target_link_libraries(multi_block_kernel_launcher
    PRIVATE
        multi_block_kernel
        CUDA::cudart
        cuda_headers
        kernel_launch_params
        my_default_flags
)