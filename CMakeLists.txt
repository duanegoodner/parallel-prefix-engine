cmake_minimum_required(VERSION 3.20)
project(PrefixSumCompare CXX CUDA)

# Enable CUDA
enable_language(CUDA)


# ===========================
# ENABLE MPI
# ===========================
find_package(MPI REQUIRED)

message(STATUS "MPI CXX Compiler: ${MPI_CXX_COMPILER}")
message(STATUS "MPI Include Path: ${MPI_INCLUDE_PATH}")
message(STATUS "MPI Libraries: ${MPI_LIBRARIES}")


# ===========================
# SET C++ STANDARD
# ===========================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ===========================
# LOCATION OF EXECUTABLE
# ===========================

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# ===========================
# BUILD CONFIGURATION OPTIONS
# ===========================

option(RESET_DEFAULTS "Reset all options to default settings" OFF)
option(ENABLE_DEBUG "Enable Debugging Symbols (-O0 -g)" OFF)
option(ENABLE_NO_INLINE "Disable Function Inlining (-fno-inline)" OFF)
option(ENABLE_LOW_OPT "Use Lower Optimization (-O1 instead of -O3)" OFF)
option(ENABLE_PROFILING "Enable Profiling (-pg, for gprof/perf)" OFF)
option(ENABLE_CONCEPT_ERROR_DETAIL "Enable Verbose Compiler Errors for Concepts" ON)

# ===========================
# HANDLE RESET_DEFAULTS OPTION
# ===========================

if (RESET_DEFAULTS)
    message(STATUS "Resetting all options to default settings (turning them OFF)...")
    
    set(ENABLE_DEBUG OFF CACHE BOOL "Disable Debugging Symbols (Reset to Default)" FORCE)
    set(ENABLE_NO_INLINE OFF CACHE BOOL "Allow Function Inlining (Reset to Default)" FORCE)
    set(ENABLE_LOW_OPT OFF CACHE BOOL "Use Default Optimization (-O3, Reset)" FORCE)
    set(ENABLE_PROFILING OFF CACHE BOOL "Disable Profiling (Reset to Default)" FORCE)
    set(ENABLE_CONCEPT_ERROR_DETAIL ON CACHE BOOL "Enable Verbose Compiler Errors for Concepts (Default)" FORCE)
endif()

# ===========================
# APPLY COMPILER OPTIONS BASED ON CONFIGURATION
# ===========================

if (ENABLE_DEBUG)
    add_compile_options(-O0 -g)
elseif (ENABLE_LOW_OPT)
    add_compile_options(-O1 -march=native)
else()
    add_compile_options(-O3 -march=native) # Default optimized build
endif()

# Apply function inlining disable
if (ENABLE_NO_INLINE)
    add_compile_options(-fno-inline)
endif()

# Allow profiling 
if (ENABLE_PROFILING)
    add_compile_options(-pg)
endif()

# Enable verbose error messages for C++ Concepts (if using GCC)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND ENABLE_CONCEPT_ERROR_DETAIL)
    add_compile_options(-fconcepts-diagnostics-depth=5)
endif()

message(STATUS "Building with compile options: ${CMAKE_CXX_FLAGS}")


# ===========================
# FETCH DEPENDENCIES
# ===========================


include(FetchContent)
include(CMakeFindDependencyMacro)

FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG        v2.3.2  # Stable version
)
FetchContent_MakeAvailable(CLI11)

FetchContent_Declare(
        googletest
        URL
        https://github.com/google/googletest/archive/ec25eea8f8237cf86c30703f59747e42f34b6f75.zip)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)



# ===========================
# INCLUDE DIRECTORIES WITH TARGETS
# ===========================
add_subdirectory(src/common)
add_subdirectory(src/mpi_prefix_sum)
add_subdirectory(src/cuda_prefix_sum)

add_subdirectory(test)





# # ===========================
# # BUILD TARGET
# # ===========================

# set (SRC_FILES
#     src/common/main.cpp
#     src/common/program_args.cpp
#     src/mpi_prefix_sum/mpi_cartesian_grid.cpp
#     src/mpi_prefix_sum/mpi_environment.cpp
#     src/mpi_prefix_sum/mpi_prefix_sum_solver.cpp
#     src/mpi_prefix_sum/mpi_utils.cpp
#     src/mpi_prefix_sum/prefix_sum_block_matrix.cpp
#     src/mpi_prefix_sum/block_matrix_mpi_distributor.cpp
# )

# add_executable(prefix_sum ${SRC_FILES})


# # ===========================
# # INCLUDE DIRECTORIES
# # ===========================

# target_include_directories(prefix_sum PRIVATE include ${MPI_INCLUDE_PATH})


# # ===========================
# # LINK MPI LIBRARY (Fix Undefined References)
# # ===========================
# target_link_libraries(prefix_sum PRIVATE MPI::MPI_CXX)
# target_link_libraries(prefix_sum PRIVATE CLI11::CLI11)


