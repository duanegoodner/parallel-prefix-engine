add_library(edge_buffer_row_kernels edge_buffer_row_kernels.cu)

target_include_directories(
    edge_buffer_row_kernels
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(edge_buffer_row_kernels
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(edge_buffer_row_kernels
    PRIVATE
        CUDA::cudart
        cuda_headers
        my_default_flags
)