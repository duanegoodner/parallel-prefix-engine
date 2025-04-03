# Set output directory for all test targets
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(TEST_RUNTIME_OUTPUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test)

# Apply the output directory to all test targets added in this scope and below
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIR})

add_subdirectory(cuda_prefix_sum)
add_subdirectory(common)
add_subdirectory(mpi_prefix_sum)
