add_subdirectory(multi_tile_kernel_launcher)
add_subdirectory(single_tile_kernel_launcher)


# Target for kernel launcher selector
add_library(cuda_kernel_launcher_selector STATIC
    cuda_kernel_launcher_selector.cpp
)

target_include_directories(cuda_kernel_launcher_selector
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(cuda_kernel_launcher_selector
    PRIVATE
        my_default_flags
        single_tile_kernel_launcher
        multi_tile_kernel_launcher
        CUDA::cudart                       
        cuda_headers
)