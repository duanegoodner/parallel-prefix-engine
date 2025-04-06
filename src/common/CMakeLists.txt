# src/common/CMakeLists.txt

add_library(prefix_sum_common
    program_args.cpp
)

target_include_directories(prefix_sum_common PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(prefix_sum_common PUBLIC CLI11::CLI11)

# Executable that links in all necessary components
# add_executable(prefix_sum main.cpp)

# target_link_libraries(prefix_sum
#     PRIVATE
#         prefix_sum_common
#         prefix_sum_mpi
#         prefix_sum_cuda
#         MPI::MPI_CXX
# )
