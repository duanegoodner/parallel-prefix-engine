add_subdirectory(mpi_cartesian_grid)
add_subdirectory(mpi_environment)
add_subdirectory(mpi_tile_info_distributor)
add_subdirectory(prefix_sum_block_matrix)


# ===========================
# MPI_PREFIX_SUM_SOLVER
# ===========================
add_executable(
    test_mpi_prefix_sum_solver
    test_mpi_prefix_sum_solver.cpp
    )
target_include_directories(
    test_mpi_prefix_sum_solver PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )
target_link_libraries(
    test_mpi_prefix_sum_solver
    PRIVATE
    GTest::gtest_main
    program_args
    prefix_sum_mpi
    MPI::MPI_CXX
    )
target_link_libraries(test_mpi_prefix_sum_solver PRIVATE my_default_flags)

