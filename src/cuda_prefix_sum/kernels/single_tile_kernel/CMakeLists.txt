add_library(
    single_tile_kernel
    single_tile_kernel.cu
    )

target_include_directories(
    single_tile_kernel
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(single_tile_kernel
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(single_tile_kernel
    PRIVATE
        CUDA::cudart
        cuda_headers
        kernel_launch_params
        my_default_flags
)

