add_library(
    multi_block_kernel
    multi_block_kernel.cu
    )

target_include_directories(
    multi_block_kernel
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(multi_block_kernel
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(multi_block_kernel
    PRIVATE
        CUDA::cudart
        cuda_headers
        kernel_launch_params
        my_default_flags
)
