// ----------------------------------------------------------------------------
// matrix_output.hpp
//
// Declares functions to format and print local matrices and distributed
// matrices.
// ----------------------------------------------------------------------------

#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

template <typename T>
std::string
FormatMatrix(int rank, int num_rows, int num_cols, const std::vector<T> &mat) {
  std::ostringstream oss;
  oss << "rank " << rank << ":";
  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_cols; ++j) {
      oss << "\t" << mat[i * num_cols + j];
    }
    oss << "\n";
  }
  return oss.str();
}

inline void PrintMatrix(const std::string &output) {
  std::cout << output << std::endl;
}