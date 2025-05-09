add_library(
    mpi_data_transfer
    mpi_cartesian_grid.cpp
    mpi_tile_info_distributor.cpp
    )

target_include_directories(mpi_data_transfer PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(
    mpi_data_transfer PUBLIC
    mpi_data_structs
    program_args
    MPI::MPI_CXX)
target_link_libraries(mpi_data_structs PRIVATE my_default_flags)