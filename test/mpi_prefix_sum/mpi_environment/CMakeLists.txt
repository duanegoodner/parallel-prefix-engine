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
    ${PROJECT_SOURCE_DIR}/src/mpi_prefix_sum/include
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

gtest_discover_tests(test_mpi_environment)