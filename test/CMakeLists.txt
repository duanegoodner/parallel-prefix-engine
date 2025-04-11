# Set output directory for all test targets
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(TEST_RUNTIME_OUTPUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test)

# Apply the output directory to all test targets added in this scope and below
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIR})

add_subdirectory(cuda_prefix_sum)
add_subdirectory(common)
add_subdirectory(mpi_prefix_sum)


enable_testing()

add_test(NAME MatrixInit COMMAND test_matrix_init)
add_test(NAME ProgramArgs COMMAND test_program_args)
add_test(NAME MPICartesianGrid COMMAND test_mpi_cartesian_grid)
add_test(NAME MPIPrefixSumSolver COMMAND test_mpi_prefix_sum_solver)
add_test(NAME MPITileDistributor COMMAND test_mpi_tile_info_distributor)
add_test(NAME MPIUtils COMMAND test_mpi_utils)
add_test(NAME MPIEnvironment COMMAND test_mpi_environment)
add_test(NAME CUDAPrefixSumSolver COMMAND test_cuda_prefix_sum_solver)
