#!/bin/bash

set -e

TEST_DIR="test/common"
TESTS=("test_matrix_init" "test_program_args")

for test in "${TESTS[@]}"; do
  mkdir -p "${TEST_DIR}/${test}"
  git mv "${TEST_DIR}/${test}.cpp" "${TEST_DIR}/${test}/${test}.cpp"

  cat > "${TEST_DIR}/${test}/CMakeLists.txt" <<EOF
add_executable(${test}
    ${test}.cpp
)

target_include_directories(${test}
    PRIVATE
    \${CMAKE_SOURCE_DIR}/include
    \${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(${test}
    PRIVATE
    GTest::gtest_main
    my_default_flags
    program_args
)

# Additional link dependencies
EOF

  # Add extra target libs if needed
  if [[ "$test" == "test_program_args" ]]; then
    echo "target_link_libraries(${test} PRIVATE prefix_sum_cuda prefix_sum_mpi)" >> "${TEST_DIR}/${test}/CMakeLists.txt"
  fi

  echo "gtest_discover_tests(${test})" >> "${TEST_DIR}/${test}/CMakeLists.txt"
done

# Replace top-level test/common/CMakeLists.txt
cat > "${TEST_DIR}/CMakeLists.txt" <<EOF
add_subdirectory(test_matrix_init)
add_subdirectory(test_program_args)
EOF

echo "✅ Test refactor complete. Re-run CMake and tests."
