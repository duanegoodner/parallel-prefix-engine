cmake_minimum_required(VERSION 3.20)
project(PrefixSumCompare CXX CUDA)

include(CTest)

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
option(ENABLE_O0 "No compiler optimization -O0" OFF)
option(ENABLE_O1 "Compiler optimization -O1" OFF)
option(ENABLE_O2 "Compiler optimization -O2" OFF)
option(ENABLE_O3 "Compiler optimization -O3" OFF)
option(ENABLE_DEBUG "Enable Debugging Symbols -g" OFF)
option(ENABLE_NO_INLINE "Disable Function Inlining -fno-inline" OFF)
option(ENABLE_PROFILING "Enable Profiling -p (for gprof)" OFF)
option(ENABLE_CONCEPT_ERROR_DETAIL "Enable Verbose Compiler Errors for Concepts" OFF)
option(ENABLE_CUDA_FAST_MATH "Enable CUDA --use_fast_math" OFF)
option(ENABLE_CUDA_DEVICE_DEBUG "Enable CUDA debug symbols (--device-debug)" OFF)


# ===========================
# HANDLE RESET_DEFAULTS OPTION
# ===========================

if (RESET_DEFAULTS)
    message(STATUS "Resetting all options to default settings (turning them OFF)...")
    set(ENABLE_O0 OFF CACHE BOOL "Disable -O0" FORCE)
    set(ENABLE_O1 OFF CACHE BOOL "Disable -O1" FORCE)
    set(ENABLE_O2 OFF CACHE BOOL "Disable -O2" FORCE)
    set(ENABLE_O3 OFF CACHE BOOL "Disable -O3" FORCE)
    set(ENABLE_DEBUG OFF CACHE BOOL "Disable Debugging Symbols (Reset to Default)" FORCE)
    set(ENABLE_NO_INLINE OFF CACHE BOOL "Allow Function Inlining (Reset to Default)" FORCE)
    set(ENABLE_PROFILING OFF CACHE BOOL "Disable Profiling (Reset to Default)" FORCE)
    set(ENABLE_CONCEPT_ERROR_DETAIL OFF CACHE BOOL "Enable Verbose Compiler Errors for Concepts (Default)" FORCE)
    set(ENABLE_CUDA_FAST_MATH OFF CACHE BOOL "Disable --use_fast_math for CUDA" FORCE)
    set(ENABLE_CUDA_DEVICE_DEBUG OFF CACHE BOOL "Disable --device-debug for CUDA" FORCE)
endif()

# ===========================
# CREATE INTERFACE TARGET FOR DEFAULT FLAGS
# ===========================

add_library(my_default_flags INTERFACE)

if (ENABLE_O0)
    target_compile_options(my_default_flags INTERFACE -O0 -march=native)
elseif (ENABLE_O1)
    target_compile_options(my_default_flags INTERFACE -O1 -march=native)
elseif (ENABLE_O2)
    target_compile_options(my_default_flags INTERFACE -O2 -march=native)
else()
    target_compile_options(my_default_flags INTERFACE -O3 -march=native) # Default optimized build
endif()

if (ENABLE_DEBUG)
    target_compile_options(my_default_flags INTERFACE -g)
endif()

# Apply function inlining disable
if (ENABLE_NO_INLINE)
    target_compile_options(my_default_flags INTERFACE -fno-inline)
endif()

# Enable verbose error messages for C++ Concepts (if using GCC)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND ENABLE_CONCEPT_ERROR_DETAIL)
    target_compile_options(my_default_flags INTERFACE -fconcepts-diagnostics-depth=5)
endif()

# Apply CUDA-specific fast math options (if applicable)
if (ENABLE_CUDA_FAST_MATH)
    target_compile_options(my_default_flags INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()

# Apply CUDA device debug flag if requested
if (ENABLE_CUDA_DEVICE_DEBUG)
    target_compile_options(my_default_flags INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--device-debug>)
endif()


get_target_property(MY_DEFAULT_COMPILER_OPTIONS my_default_flags INTERFACE_COMPILE_OPTIONS)
message(STATUS "Compile options for my_default_flags: ${MY_DEFAULT_COMPILER_OPTIONS}")


# ===========================
# APPLY COMPILER OPTIONS BASED ON CONFIGURATION
# ===========================

# # Create an empty variable to track flags
# set(MY_COMPILE_FLAGS "")

# if (ENABLE_O0)
#     add_compile_options(-O0 -march=native)
#     set(MY_COMPILE_FLAGS "${MY_COMPILE_FLAGS} -O0 -march=native")
# elseif (ENABLE_O1)
#     add_compile_options(-O1 -march=native)
#     set(MY_COMPILE_FLAGS "${MY_COMPILE_FLAGS} -O1 -march=native")
# elseif (ENABLE_O2)
#     add_compile_options(-O2 -march=native)
#     set(MY_COMPILE_FLAGS "${MY_COMPILE_FLAGS} -O2 -march=native")
# else()
#     add_compile_options(-O3 -march=native) # Default optimized build
#     set(MY_COMPILE_FLAGS "${MY_COMPILE_FLAGS} -O3 -march=native")
# endif()

# if (ENABLE_DEBUG)
#     add_compile_options(-g)
#     set(MY_COMPILE_FLAGS "${MY_COMPILE_FLAGS} -g")
# endif()

# # Apply function inlining disable
# if (ENABLE_NO_INLINE)
#     add_compile_options(-fno-inline)
#     set(MY_COMPILE_FLAGS "${MY_COMPILE_FLAGS} -fno-inline")
# endif()

# # Enable verbose error messages for C++ Concepts (if using GCC)
# if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND ENABLE_CONCEPT_ERROR_DETAIL)
#     add_compile_options(-fconcepts-diagnostics-depth=5)
#     set(MY_COMPILE_FLAGS "${MY_COMPILE_FLAGS} -fconcepts-diagnostics-depth=5")
# endif()


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



