add_library(cuda_solver_registration STATIC
    cuda_solver_registration.cpp
)

target_include_directories(cuda_solver_registration
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(cuda_solver_registration
    PRIVATE
        cuda_kernel_launcher_selector
        CUDA::cudart
        cuda_headers  # if you use it inside the registration code
)
