add_library(
    sub_tile_kernels
    sub_tile_kernels.cu
    )

target_include_directories(
    sub_tile_kernels
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(sub_tile_kernels
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(sub_tile_kernels
    PRIVATE
        CUDA::cudart
        cuda_headers
        kernel_launch_params
        my_default_flags
)