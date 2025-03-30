#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

template <typename T>
std::string FormatMatrix(int rank, int local_n, const std::vector<T> &mat) {
  std::ostringstream oss;
  oss << "rank " << rank << ":";
  for (int i = 0; i < local_n; ++i) {
    for (int j = 0; j < local_n; ++j) {
      oss << "\t" << mat[i * local_n + j];
    }
    oss << "\n";
  }
  return oss.str();
}

inline void PrintMatrix(const std::string &output) {
  std::cout << output << std::endl;
}
