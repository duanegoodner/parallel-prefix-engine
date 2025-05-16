# src/mpi_prefix_sum/CMakeLists.txt

add_subdirectory(mpi_data_structs)
add_subdirectory(mpi_data_transfer)
add_subdirectory(mpi_environment)
add_subdirectory(mpi_utils)
add_subdirectory(registration)


add_library(prefix_sum_mpi mpi_prefix_sum_solver.cpp)

target_include_directories(prefix_sum_mpi PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(prefix_sum_mpi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(
    prefix_sum_mpi PRIVATE
    mpi_data_structs
    mpi_data_transfer
    mpi_environment
    program_args
    time_utils
    my_default_flags
    MPI::MPI_CXX
    )
