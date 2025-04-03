# src/mpi_prefix_sum/CMakeLists.txt

add_library(prefix_sum_mpi
    mpi_cartesian_grid.cpp
    mpi_environment.cpp
    mpi_prefix_sum_solver.cpp
    mpi_utils.cpp
    prefix_sum_block_matrix.cpp
    mpi_tile_info_distributor.cpp
)

target_include_directories(prefix_sum_mpi PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(prefix_sum_mpi PUBLIC prefix_sum_common)
