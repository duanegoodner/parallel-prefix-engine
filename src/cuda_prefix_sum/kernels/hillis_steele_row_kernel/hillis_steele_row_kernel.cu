// === Single-block exclusive scan (Hillis-Steele) for a single row ===
__global__ void RowWiseScanSingleBlock(
    const int* __restrict__ in,
    int* __restrict__ out,
    int num_cols)
{
    extern __shared__ int temp[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (tid >= num_cols) return;

    // Load input to shared memory
    temp[tid] = in[row * num_cols + tid];
    __syncthreads();

    // Inclusive Hillis-Steele scan
    for (int offset = 1; offset < num_cols; offset *= 2) {
        int val = 0;
        if (tid >= offset)
            val = temp[tid - offset];
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    // Convert to exclusive scan
    if (tid == 0) {
        out[row * num_cols + tid] = 0;
    } else if (tid < num_cols) {
        out[row * num_cols + tid] = temp[tid - 1];
    }
}