#include <gtest/gtest.h>

#include <algorithm>

#include "mpi_prefix_sum/internal/prefix_sum_block_matrix.hpp"

class PrefixSumBlockMatrixTest : public ::testing::Test {};

TEST_F(PrefixSumBlockMatrixTest, DefaultInit) {
  auto block_matrix = PrefixSumBlockMatrix();
  EXPECT_EQ(block_matrix.data().size(), 0);
}

TEST_F(PrefixSumBlockMatrixTest, InitSquareZeros) {
  int square_dim = 2;
  auto block_matrix = PrefixSumBlockMatrix(square_dim);
  EXPECT_EQ(block_matrix.data().size(), square_dim * square_dim);

  auto are_all_zeros = std::all_of(
      block_matrix.data().begin(),
      block_matrix.data().end(),
      [](int value) { return value == 0; }
  );
  EXPECT_TRUE(are_all_zeros);
  EXPECT_EQ(block_matrix.num_rows(), square_dim);
  EXPECT_EQ(block_matrix.num_cols(), square_dim);
}

TEST_F(PrefixSumBlockMatrixTest, InitRectangularZeros) {
  int num_rows = 2;
  int num_cols = 3;
  auto block_matrix = PrefixSumBlockMatrix(num_rows, num_cols);
  EXPECT_EQ(block_matrix.data().size(), num_rows * num_cols);

  auto are_all_zeros = std::all_of(
      block_matrix.data().begin(),
      block_matrix.data().end(),
      [](int value) { return value == 0; }
  );
  EXPECT_TRUE(are_all_zeros);
  EXPECT_EQ(block_matrix.num_rows(), num_rows);
  EXPECT_EQ(block_matrix.num_cols(), num_cols);
}

TEST_F(PrefixSumBlockMatrixTest, InitRectangularWithData) {
  int num_rows = 2;
  int num_cols = 3;
  std::vector<int> data = {1, 2, 3, 4, 5, 6};
  auto block_matrix =
      PrefixSumBlockMatrix(num_rows, num_cols, std::move(data));
  EXPECT_EQ(block_matrix.data().size(), num_rows * num_cols);

  EXPECT_EQ(block_matrix.num_rows(), num_rows);
  EXPECT_EQ(block_matrix.num_cols(), num_cols);

  EXPECT_EQ(block_matrix.ValueAt(0, 0), 1);
  EXPECT_EQ(block_matrix.ValueAt(0, 1), 2);
  EXPECT_EQ(block_matrix.ValueAt(0, 2), 3);
  EXPECT_EQ(block_matrix.ValueAt(1, 0), 4);
  EXPECT_EQ(block_matrix.ValueAt(1, 1), 5);
  EXPECT_EQ(block_matrix.ValueAt(1, 2), 6);

  EXPECT_EQ(data.size(), 0);
}

TEST_F(PrefixSumBlockMatrixTest, ComputeLocalPrefixSum) {
  int num_rows = 2;
  int num_cols = 3;
  std::vector<int> data = {1, 2, 3, 4, 5, 6};
  auto block_matrix =
      PrefixSumBlockMatrix(num_rows, num_cols, std::move(data));

  block_matrix.ComputeLocalPrefixSum();

  EXPECT_EQ(block_matrix.ValueAt(0, 0), 1);
  EXPECT_EQ(block_matrix.ValueAt(0, 1), 3);
  EXPECT_EQ(block_matrix.ValueAt(0, 2), 6);
  EXPECT_EQ(block_matrix.ValueAt(1, 0), 5);
  EXPECT_EQ(block_matrix.ValueAt(1, 1), 12);
  EXPECT_EQ(block_matrix.ValueAt(1, 2), 21);
}

TEST_F(PrefixSumBlockMatrixTest, SubDivide) {
  int num_rows = 6;
  int num_cols = 6;
  std::vector<int> data = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                           13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                           25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36};
  auto block_matrix =
      PrefixSumBlockMatrix(num_rows, num_cols, std::move(data));

  int tiles_per_row = 2;
  int tiles_per_col = 2;

  auto sub_matrices = block_matrix.SubDivide(tiles_per_row, tiles_per_col);

  EXPECT_EQ(sub_matrices.size(), tiles_per_row * tiles_per_col);

  int rows_per_tile = num_rows / tiles_per_row;
  int cols_per_tile = num_cols / tiles_per_col;

  for (auto &sub_matrix : sub_matrices) {
    EXPECT_EQ(sub_matrix.second.size(), rows_per_tile * cols_per_tile);
  }

  EXPECT_EQ(sub_matrices[0][0], 1);
  EXPECT_EQ(sub_matrices[0][1], 2);
  EXPECT_EQ(sub_matrices[0][2], 3);
  EXPECT_EQ(sub_matrices[0][3], 7);
  EXPECT_EQ(sub_matrices[0][4], 8);
  EXPECT_EQ(sub_matrices[0][5], 9);
  EXPECT_EQ(sub_matrices[0][6], 13);
  EXPECT_EQ(sub_matrices[0][7], 14);
  EXPECT_EQ(sub_matrices[0][8], 15);
}

TEST_F(PrefixSumBlockMatrixTest, Print) {
  int num_rows = 6;
  int num_cols = 6;
  std::vector<int> data = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                           13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                           25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36};

  auto block_matrix =
      PrefixSumBlockMatrix(num_rows, num_cols, std::move(data));
  block_matrix.Print();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}