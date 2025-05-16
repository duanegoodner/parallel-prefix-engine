add_library(
    multi_tile_kernel
    multi_tile_kernel.cu
    )

target_include_directories(
    multi_tile_kernel
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(multi_tile_kernel
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(multi_tile_kernel
    PRIVATE
        CUDA::cudart
        cuda_headers
        kernel_launch_params
        my_default_flags
)
