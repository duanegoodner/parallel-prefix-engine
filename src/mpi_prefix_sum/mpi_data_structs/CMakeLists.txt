add_library(mpi_data_structs prefix_sum_block_matrix.cpp)

target_include_directories(mpi_data_structs PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(mpi_data_structs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(mpi_data_structs PUBLIC MPI::MPI_CXX)
target_link_libraries(mpi_data_structs PRIVATE my_default_flags)