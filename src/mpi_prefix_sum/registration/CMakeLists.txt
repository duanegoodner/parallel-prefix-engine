add_library(mpi_solver_registration STATIC
    mpi_solver_registration.cpp
)

target_include_directories(mpi_solver_registration
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(mpi_solver_registration
    PRIVATE
        MPI::MPI_CXX
)