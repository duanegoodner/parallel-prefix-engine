add_library(mpi_environment mpi_environment.cpp)

target_include_directories(mpi_environment PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(mpi_environment PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(mpi_environment PUBLIC  MPI::MPI_CXX)
target_link_libraries(mpi_environment PRIVATE time_utils my_default_flags)