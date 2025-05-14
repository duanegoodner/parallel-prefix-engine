cmake_minimum_required(VERSION 3.20)
project(PrefixSumCompare CXX CUDA)



# ===========================
# ENABLE TESTING
# ===========================
include(CTest)
enable_testing()
include(GoogleTest)
# NOTE: Also need to call gtest_discover_tests(<test-executable>)
# for every test compilation target. Currently doing this in 
# each CMakeLists.txt file that defines a test target.


# ===========================
# ENABLE CUDA
# ===========================
find_package(CUDAToolkit REQUIRED)

# Define the shared interface target
add_library(cuda_headers INTERFACE)
target_include_directories(cuda_headers INTERFACE ${CUDAToolkit_INCLUDE_DIRS})


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


# Optimization levels (only one should be ON)
option(ENABLE_O0 "No compiler optimization -O0" OFF)
option(ENABLE_O1 "Compiler optimization -O1" OFF)
option(ENABLE_O2 "Compiler optimization -O2" OFF)
option(ENABLE_O3 "Compiler optimization -O3" OFF)  # Default if none set

# General debugging (applies -g -O0 for CXX, -g -G -O0 for CUDA)
option(ENABLE_DEBUG "Enable debug symbols and disable optimization" OFF)

# Optional performance/debugging toggles
option(ENABLE_PROFILING "Enable profiling with -p (for gprof)" OFF)
option(ENABLE_CUDA_FAST_MATH "Enable CUDA --use_fast_math (not recommended for debug builds)" OFF)


# ===========================
# CREATE INTERFACE TARGET FOR DEFAULT FLAGS
# ===========================
add_library(my_default_flags INTERFACE)

# Apply debug or optimization flags
if (ENABLE_DEBUG)
    message(STATUS "Debugging is ENABLED: -g -O0 for host, -g -G -O0 for CUDA")
    target_compile_options(my_default_flags INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-g -O0 -march=native -fno-omit-frame-pointer>
        $<$<COMPILE_LANGUAGE:CUDA>:-G;-g;-O0;-lineinfo>
    )
else()
    # Apply optimization level (default is -O3)
    if (ENABLE_O0)
        target_compile_options(my_default_flags INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-O0 -march=native>
            $<$<COMPILE_LANGUAGE:CUDA>:-O0>
        )
    elseif (ENABLE_O1)
        target_compile_options(my_default_flags INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-O1 -march=native>
            $<$<COMPILE_LANGUAGE:CUDA>:-O1>
        )
    elseif (ENABLE_O2)
        target_compile_options(my_default_flags INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-O2 -march=native>
            $<$<COMPILE_LANGUAGE:CUDA>:-O2>
        )
    else()
        target_compile_options(my_default_flags INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        )
    endif()
endif()

# Apply gprof profiling flag
if (ENABLE_PROFILING)
    message(STATUS "Profiling is ENABLED: Adding -pg")
    target_compile_options(my_default_flags INTERFACE -pg)
    target_link_options(my_default_flags INTERFACE -pg)
endif()

# CUDA fast math (only for performance builds)
if (ENABLE_CUDA_FAST_MATH AND NOT ENABLE_DEBUG)
    target_compile_options(my_default_flags INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
endif()

# Show final compile options at configure time
get_target_property(MY_DEFAULT_COMPILER_OPTIONS my_default_flags INTERFACE_COMPILE_OPTIONS)
message(STATUS "Compile options for my_default_flags: ${MY_DEFAULT_COMPILER_OPTIONS}")



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

add_subdirectory(src)
add_subdirectory(test)



