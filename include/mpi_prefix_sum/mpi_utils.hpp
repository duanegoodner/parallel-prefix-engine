#pragma once

#include <cassert>
#include <cstddef>
#include <iostream>
#include <vector>

void print_local_mat(int rank, int local_n, const std::vector<int> &local_mat);
void print_global_mat(
    int rank,
    int procs,
    int local_n,
    const std::vector<int> &local_mat
);

template <typename T>
inline T &
value_at(std::vector<T> &array, int row_idx, int col_idx, int stride) {
  assert(
      row_idx >= 0 && col_idx >= 0 &&
      static_cast<size_t>(row_idx * stride + col_idx) < array.size()
  );
  return array[row_idx * stride + col_idx];
}

template <typename T>
inline const T &
value_at(const std::vector<T> &array, int row_idx, int col_idx, int stride) {
  assert(
      row_idx >= 0 && col_idx >= 0 &&
      static_cast<size_t>(row_idx * stride + col_idx) < array.size()
  );
  return array[row_idx * stride + col_idx];
}

struct ProgramArgs {
  int local_n;
  int seed;
};

ProgramArgs get_args(int argc, char *argv[], int rank);
