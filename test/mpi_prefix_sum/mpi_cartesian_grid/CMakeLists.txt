# ===========================
# MPI_CARTESIAN_GRID
# ===========================
add_executable(
    test_mpi_cartesian_grid
    test_mpi_cartesian_grid.cpp
    )
target_include_directories(
    test_mpi_cartesian_grid PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/mpi_prefix_sum/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )
target_link_libraries(
    test_mpi_cartesian_grid
    PRIVATE
    GTest::gtest_main
    program_args
    prefix_sum_mpi
    MPI::MPI_CXX
    )
target_link_libraries(test_mpi_cartesian_grid PRIVATE my_default_flags)