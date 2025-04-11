# src/common/CMakeLists.txt

add_library(prefix_sum_common
    program_args.cpp
)
target_include_directories(prefix_sum_common PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(prefix_sum_common PRIVATE my_default_flags)

add_library(arg_parser arg_parser.cpp)
target_include_directories(arg_parser PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(arg_parser PUBLIC CLI11::CLI11)
target_link_libraries(arg_parser PRIVATE my_default_flags)


add_library(solver_dispatch solver_dispatch.cpp)
target_include_directories(solver_dispatch PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(solver_dispatch PUBLIC MPI::MPI_CXX)
target_link_libraries(solver_dispatch PRIVATE my_default_flags)


# Executable that links in all necessary components
add_executable(prefix_sum main.cpp)
target_link_libraries(prefix_sum
    PRIVATE
        arg_parser
        solver_dispatch
        prefix_sum_common
        prefix_sum_mpi
        prefix_sum_cuda
        MPI::MPI_CXX
)
target_link_libraries(arg_parser PRIVATE my_default_flags)
