add_subdirectory(common)
add_subdirectory(cuda_prefix_sum)
add_subdirectory(mpi_prefix_sum)
# add_subdirectory(zscaffold_prefix_sum)

# Executable that links in all necessary components
add_executable(prefix_sum main.cpp)
target_link_libraries(prefix_sum
    PRIVATE
        arg_parser
        solver_dispatch
        time_utils
        program_args
        prefix_sum_mpi
        prefix_sum_cuda
        # prefix_sum_zscaffold
        MPI::MPI_CXX
)
target_link_libraries(arg_parser PRIVATE my_default_flags)