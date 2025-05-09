add_library(
    subtile_kernel
    subtile_kernel.cu
    )

target_include_directories(
    subtile_kernel
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(subtile_kernel
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(subtile_kernel
    PRIVATE
        CUDA::cudart
        cuda_headers
        kernel_launch_params
        my_default_flags
)

