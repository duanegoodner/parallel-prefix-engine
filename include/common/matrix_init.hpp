// ----------------------------------------------------------------------------
// matrix_init.hpp
//
// Provides a templated function to generate a square matrix of random values,
// useful for both integral and floating point types. Used in test drivers.
// ----------------------------------------------------------------------------

#pragma once

#include <random>
#include <type_traits>
#include <vector>

template <typename T>
std::vector<T> GenerateRandomMatrix(
    int num_rows,
    int num_cols,
    int seed,
    T low = T(-10),
    T high = T(10)
) {
  std::mt19937 rng(seed);

  std::vector<T> mat(num_rows * num_cols);

  if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<T> dist(low, high - 1);
    for (T &val : mat)
      val = dist(rng);
  } else if constexpr (std::is_floating_point_v<T>) {
    std::uniform_real_distribution<T> dist(low, high);
    for (T &val : mat)
      val = dist(rng);
  } else {
    static_assert(
        std::is_arithmetic_v<T>,
        "GenerateRandomMatrix requires an arithmetic type."
    );
  }

  return mat;
}