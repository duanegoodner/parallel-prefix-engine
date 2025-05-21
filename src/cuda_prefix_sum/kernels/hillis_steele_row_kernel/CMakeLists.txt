add_library(hillis_steele_row_kernel hillis_steele_row_kernel.cu)

target_include_directories(
    hillis_steele_row_kernel
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(hillis_steele_row_kernel
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(hillis_steele_row_kernel
    PRIVATE
        CUDA::cudart
        cuda_headers
        my_default_flags
)