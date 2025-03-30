#pragma once

#include <cassert>
#include <cstddef>
#include <iostream>
#include <vector>

// void PrintLocalMat(int rank, int local_n, const std::vector<int> &local_mat);
// void PrintGlobalMat(
//     int rank,
//     int procs,
//     int local_n,
//     const std::vector<int> &local_mat
// );

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

class ProgramArgs {
private:
  int local_n_ = 0;
  int seed_ = 1234;

public:
  ProgramArgs() = default;

  ProgramArgs(int local_n, int seed)
      : local_n_(local_n)
      , seed_(seed) {}

  static ProgramArgs Parse(int argc, char *argv[], int rank);

  int local_n() { return local_n_; }
  void set_local_n(int value) { local_n_ = value; }

  int seed() { return seed_; }
  void set_seed(int value) { seed_ = value; }
};
