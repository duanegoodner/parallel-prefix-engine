add_library(solver_dispatch solver_dispatch.cpp)
target_include_directories(solver_dispatch PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(solver_dispatch PUBLIC cuda_headers MPI::MPI_CXX)
target_link_libraries(
    solver_dispatch
    PRIVATE
    my_default_flags
    multi_block_kernel_launcher
    subtile_kernel_launcher
    prefix_sum_mpi
    )