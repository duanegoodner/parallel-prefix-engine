add_library(row_to_col_injection_kernel row_to_col_injection.cu)

target_include_directories(
    row_to_col_injection_kernel
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(row_to_col_injection_kernel
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(row_to_col_injection_kernel
    PRIVATE
        CUDA::cudart
        cuda_headers
        my_default_flags
)