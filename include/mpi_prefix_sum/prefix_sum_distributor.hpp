#ifndef MPI_PREFIX_SUM_PREFIX_SUM_DISTRIBUTOR_H_
#define MPI_PREFIX_SUM_PREFIX_SUM_DISTRIBUTOR_H_

#include "prefix_sum_block_matrix.hpp"
#include <mpi.h>

class PrefixSumDistributor {
 public:
  PrefixSumDistributor(PrefixSumBlockMatrix& matrix, int proc_row, int proc_col, int p);

  void Distribute(MPI_Comm comm_row, MPI_Comm comm_col);

 private:
  void BroadcastRowPrefixSums(MPI_Comm row_comm);
  void BroadcastColPrefixSums(MPI_Comm col_comm);

  PrefixSumBlockMatrix& matrix_;
  int proc_row_;
  int proc_col_;
  int p_;
};

#endif  // MPI_PREFIX_SUM_PREFIX_SUM_DISTRIBUTOR_H_