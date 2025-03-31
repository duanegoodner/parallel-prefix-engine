#pragma once

#include <cassert>
#include <cstddef>
#include <iostream>
#include <vector>

template <typename T>
inline T &
ValueAt(std::vector<T> &array, int row_idx, int col_idx, int stride) {
  assert(
      row_idx >= 0 && col_idx >= 0 &&
      static_cast<size_t>(row_idx * stride + col_idx) < array.size()
  );
  return array[row_idx * stride + col_idx];
}

template <typename T>
inline const T &
ValueAt(const std::vector<T> &array, int row_idx, int col_idx, int stride) {
  assert(
      row_idx >= 0 && col_idx >= 0 &&
      static_cast<size_t>(row_idx * stride + col_idx) < array.size()
  );
  return array[row_idx * stride + col_idx];
} 
