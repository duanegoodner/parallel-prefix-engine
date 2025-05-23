add_library(col_scan_single_block_kernel col_scan_single_block.cu)

target_include_directories(
    col_scan_single_block_kernel
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(col_scan_single_block_kernel
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(col_scan_single_block_kernel
    PRIVATE
        CUDA::cudart
        cuda_headers
        my_default_flags
)