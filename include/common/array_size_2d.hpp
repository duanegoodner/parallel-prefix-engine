#pragma once

struct ArraySize2D {
  size_t num_rows;
  size_t num_cols;
};

// Non-member operator==
inline bool operator==(const ArraySize2D& lhs, const ArraySize2D& rhs) {
  return lhs.num_rows == rhs.num_rows && lhs.num_cols == rhs.num_cols;
}

inline bool operator!=(const ArraySize2D& lhs, const ArraySize2D& rhs) {
  return !(lhs == rhs);
}