add_library(single_tile_kernel_launcher single_tile_kernel_launcher.cu)

target_include_directories(
    single_tile_kernel_launcher
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(single_tile_kernel_launcher
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)


target_link_libraries(single_tile_kernel_launcher
    PRIVATE
        sub_tile_kernels
        CUDA::cudart
        cuda_headers
        kernel_launch_params
        my_default_flags
)