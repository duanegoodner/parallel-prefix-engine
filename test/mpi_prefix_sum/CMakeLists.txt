
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


# ===========================
# MPI_ENVIRONMENT
# ===========================
add_executable(
    test_mpi_environment
    test_mpi_environment.cpp
    )
target_include_directories(
    test_mpi_environment PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )
target_link_libraries(
    test_mpi_environment
    PRIVATE
    GTest::gtest_main
    program_args
    prefix_sum_mpi
    MPI::MPI_CXX
    )
target_link_libraries(test_mpi_environment PRIVATE my_default_flags)


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


# ===========================
# MPI_TILE_INFO_DISTRIBUTOR
# ===========================
add_executable(
    test_mpi_tile_info_distributor
    test_mpi_tile_info_distributor.cpp
    )
target_include_directories(
    test_mpi_tile_info_distributor PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )
target_link_libraries(
    test_mpi_tile_info_distributor
    PRIVATE
    GTest::gtest_main
    program_args
    prefix_sum_mpi
    MPI::MPI_CXX
    )
target_link_libraries(test_mpi_tile_info_distributor PRIVATE my_default_flags)


# ===========================
# MPI_UTILS
# ===========================
add_executable(
    test_mpi_utils
    test_mpi_utils.cpp
    )
target_include_directories(
    test_mpi_utils PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    )
target_link_libraries(
    test_mpi_utils
    PRIVATE
    GTest::gtest_main
    program_args
    prefix_sum_mpi
    )
target_link_libraries(test_mpi_utils PRIVATE my_default_flags)


# ===========================
# PREFIX_SUM_BLOCK_MATRIX
# ===========================

add_executable(
    test_prefix_sum_block_matrix
    test_prefix_sum_block_matrix.cpp
    )
target_include_directories(
    test_prefix_sum_block_matrix PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    )
target_link_libraries(
    test_prefix_sum_block_matrix
    PRIVATE
    GTest::gtest_main
    program_args
    prefix_sum_mpi
    MPI::MPI_CXX
    )
target_link_libraries(test_prefix_sum_block_matrix PRIVATE my_default_flags)