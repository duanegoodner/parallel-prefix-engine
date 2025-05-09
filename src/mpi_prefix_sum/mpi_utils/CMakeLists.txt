add_library(mpi_utils mpi_utils.cpp)

target_include_directories(mpi_utils PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(mpi_utils PUBLIC program_args MPI::MPI_CXX)
target_link_libraries(mpi_utils PRIVATE my_default_flags)