#include <gtest/gtest.h>

#include <algorithm>

#include "mpi_prefix_sum/prefix_sum_block_matrix.hpp"

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
  auto block_matrix = PrefixSumBlockMatrix(num_rows, num_cols, std::move(data));
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

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}