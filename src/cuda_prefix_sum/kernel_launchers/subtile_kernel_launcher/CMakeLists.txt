add_library(subtile_kernel_launcher subtile_kernel_launcher.cu)

target_include_directories(
    subtile_kernel_launcher
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(subtile_kernel_launcher
    PRIVATE
        subtile_kernel
        CUDA::cudart
        cuda_headers
        kernel_launch_params
        my_default_flags
)