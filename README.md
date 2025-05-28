# Parallel Prefix Engine

***CUDA + MPI Based Framework for Parallel Data Aggregation***

## Overview

**Parallel Prefix Engine** is a C++ project that computes 2D prefix sums (also known as integral images) of integer arrays using two parallel computing approaches: **CUDA** and **MPI**.  This project began as a learning exercise and evolved into an extensible framework for experimenting with parallel computing patterns across CPU and GPU.

The current version includes:
- Flexible command-line interface for configuring runtime behavior.
- Docker-based development environment with GPU debugging support (`cuda-gdb`, Nsight), and separate containers for OpenMPI and MPICH to avoid runtime conflicts between the two MPI implementations.
- Multiple CMake build presets for optimized release builds, full-source debugging (host + CUDA), and performance profiling with `gprof` or `perf`.


## Getting Started

### Requirements

- A Nvidia GPU with drivers that support CUDA 12.9+ (preferred), although 12.2+ is generally sufficient.
- Docker
- Nvidia Container Toolkit
> [!NOTE]  
> This project uses CUDA 12.9 to ensure compatibility with the latest NVIDIA Nsight debugging capabilities in the VS Code extension. For all other aspects of the project, older CUDA versions (12.2 or later) should work fine.  To switch, replace the base image `nvidia/cuda:12.9.0-devel-ubuntu22.04` in `Dockerfile.hpc_base` with an older version (e.g. `nvidia/cuda:12.2.2-devel-ubuntu22.04`).


### Clone the Repo

```shell
git clone https://github.com/duanegoodner/parallel-prefix-engine
```

### Docker Setup
After cloning the repository, set up the Docker environment to start developing or running benchmarks.

#### Create and Edit `.env` file

```shell
cd parallel-prefix-engine/docker/parallel_env
cp .env.example .env
```
In `.env`, edit the value of `LOCAL_PROJECT_ROOT` to match the local absolute path of the project repo root.

#### Build Docker Images

```shell
UID=${UID} GID=${GID} docker compose build hpc_base 
UID=${UID} GID=${GID} docker compose build hpc_openmpi
UID=${UID} GID=${GID} docker compose build hpc_mpich
```

#### Run and Exec Into Container

In the commands below, we run and enter a `hpc_mpich` container. We could do the same with an `hpc_openmpi` container by replacing `hpc_mpich` with `hpc_openmpi`.

```shell
UID=${UID} GID=${GID} docker compose up -d hpc_mpich
docker exec -it hpc_mpich /bin/zsh
```

### Building

Once inside the container, you can configure and build the project using CMake. The example below uses the release preset for optimized builds.

```shell
cmake --preset release
cmake --build --preset release
```

### CLI Help

After building, you can use the CLI to explore different runtime options and select a backend (MPI or CUDA), matrix size, tile layout, and more.

```shell
./build/release/bin/prefix_sum --help
```
Output:
```shell
Parallel prefix sum runner
Usage: ./build/release/bin/prefix_sum [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  -r,--seed INT [1234]        Random seed
  -b,--backend TEXT:{mpi,cuda} [mpi] 
                              Backend to use (mpi or cuda)
  -L,--log-level TEXT:{off,info,warning,error} [warning] 
                              Logging level
  -f,--full-matrix-dim UINT [[4,4]]  x 2
                              Full matrix dimensions (rows cols)
  -t,--tile-dim UINT [[4,4]]  x 2
                              Tile dimensions (rows cols)
  -k,--kernel TEXT:{single_tile,multi_tile}
                              CUDA kernel type (single_tile, multi_tile)
  -s,--sub-tile-dim UINT x 2  Sub-tile dimensions (rows cols, CUDA only)
  -p,--print-full-matrix      Print the full matrix after computation

```

### Testing with Small Input Array

First, run a quick test of each back end with a small (8x8) array. Since the array is small, we can pass the `-p` option to print the full array before and after prefix sum calculation without overwhelming our terminal output.

#### CUDA Backend

For the CUDA test with an 8x8 array, we break the input into tiles of size 4x4. Each tile has its local prefix sum calculated in a separate thread block. Tiles are further subdivided into 2x2 subtiles, with a single thread handling the local prefix sum of a subtile.

```shell
./build/release/bin/prefix_sum -f 8 8 -t 4 4 -s 2 2 -b cuda -k multi_tile -p -r 1234
```
Output:
```shell
Starting matrix
-6   0	3	7	-1	2	6	6	
 6	 8	-5	-7	-5	-6	6	7	
10	-7	8	-8	-3	-10	0	0	
 4	-4	4	6	-3	-8	1	-9	
 0	-1	-10	-10	6	-4	8	-5	
-3	 5	2	8	-9	10	-3	-8	
 9	-2	3	-1	-2	1	6	6	
-4	-1	1	-1	8	1	-1	10	
After prefix sum computation
-6	-6	-3	4	3	5	11	17	
0	8	6	6	0	-4	8	21	
10	11	17	9	0	-14	-2	11	
14	11	21	19	7	-15	-2	2	
14	10	10	-2	-8	-34	-13	-14	
11	12	14	10	-5	-21	-3	-12	
20	19	24	19	2	-13	11	8	
16	14	20	14	5	-9	14	21	

=== Runtime Report ===
Total Time: 0.469147 ms
Copy to Device Time: 0.026501 ms
Device Compute Time: 0.430561 ms
Copy From Device Time: 0.011637 ms
```

#### MPI Backend
For this MPI test, we also divide the input into four tiles. The local prefix sum of each tile is calculated by a separate MPI Rank. 

```shell
mpirun -n 4 ./build/release/bin/prefix_sum -f 8 8 -t 4 4 -p -r 1234
```
Output:
```shell
Starting matrix
-6	0	3	7	-1	2	6	6	
6	8	-5	-7	-5	-6	6	7	
10	-7	8	-8	-3	-10	0	0	
4	-4	4	6	-3	-8	1	-9	
0	-1	-10	-10	6	-4	8	-5	
-3	5	2	8	-9	10	-3	-8	
9	-2	3	-1	-2	1	6	6	
-4	-1	1	-1	8	1	-1	10	
After prefix sum computation
-6	-6	-3	4	3	5	11	17	
0	8	6	6	0	-4	8	21	
10	11	17	9	0	-14	-2	11	
14	11	21	19	7	-15	-2	2	
14	10	10	-2	-8	-34	-13	-14	
11	12	14	10	-5	-21	-3	-12	
20	19	24	19	2	-13	11	8	
16	14	20	14	5	-9	14	21	


=== Runtime Report ===
Total runtime (wall clock): 0.099179 ms

=== Per-Rank Timing Breakdown (ms) ===
    Rank          Total     Distribute        Compute         Gather
--------------------------------------------------------------------
       0         0.0972         0.0138         0.0643         0.0172
       1         0.0835         0.0163         0.0645         0.0011
       2         0.0829         0.0147         0.0653         0.0010
       3         0.0781         0.0099         0.0654         0.0012
```

### Larger Input

Next, test each backend with a much larger array to observe performance scaling.

#### CUDA
```shell
./build/release/bin/prefix_sum -f 16384 16384 -t 128 128 -s 4 4 -b cuda -k multi_tile -r 1234
```
Output:
```shell
Lower right corner element after prefix sum:
52883

=== Runtime Report ===
Total Time: 163.251 ms
Copy to Device Time: 66.7048 ms
Device Compute Time: 26.0273 ms
Copy From Device Time: 70.5174 ms
```

#### MPI
```shell
mpirun -n 16 ./build/release/bin/prefix_sum -f 16384 16384 -t 4096 4096 -r 1234
```
Output:
```shell
Lower right corner element after prefix sum:
52883
=== Runtime Report ===
Total runtime (wall clock): 2749.19 ms

=== Per-Rank Timing Breakdown (ms) ===
    Rank          Total     Distribute        Compute         Gather
--------------------------------------------------------------------
       0      2749.1877      1678.6060       567.2406       503.3379
       1      2443.9813      1458.3430       773.5991       212.0366
       2      2456.2239      1475.8551       770.1155       210.2508
       3      2468.2422      1485.5764       746.4327       236.2307
       4      2480.8063      1496.4901       735.4366       248.8773
       5      2492.9617      1518.8973       727.0247       247.0366
       6      2505.7939      1543.4642       702.4930       259.8334
       7      2517.9829      1569.5674       676.3307       272.0833
       8      2532.4362      1583.3643       648.4208       300.6486
       9      2545.9569      1598.8481       633.2069       313.8996
      10      2555.1209      1632.0941       613.8332       309.1910
      11      2561.4190      1649.1909       582.8098       329.4160
      12      2568.3176      1667.4673       563.9430       336.9050
      13      2574.4284      1440.0339       791.9037       342.4877
      14      2581.0698      1432.4015       813.6587       335.0076
      15      2587.2344      1419.4285       826.5156       341.2885
```


### Results Comparison Summary

For small array sizes, the MPI backend tends to outperform CUDA due to lower overhead: data transfers between host and device dominate runtime in small workloads, and MPI's lightweight CPU threads can execute quickly across multiple cores.

However, as input size increases, the benefits of GPU parallelism and fast shared memory become apparent. For large arrays (e.g., 16K × 16K), the CUDA backend is significantly faster, particularly in compute time, even when including memory transfer overhead.

These results reflect a typical crossover point in hybrid CPU/GPU architectures: CUDA excels at high-throughput, large-scale computation, while MPI is better suited for smaller or latency-sensitive tasks.


## Build Options

Parallel Prefix Engine uses CMake presets to simplify configuration for common development and performance scenarios. Presets are defined in `CMakePresets.json` and can be invoked with `cmake --preset <name>`.

To list available presets:
```bash
cmake --list-presets
```
Available options include:  
- debug: Full debug symbols and frame pointers (including CUDA -G) for source-level debugging with tools like `cuda-gdb` and Nsight.

- release: Optimized build for benchmarks or production runs.

- release-profiling: Adds instrumentation for gprof CPU profiling.

- perf-profiling: Enables Linux perf and flamegraph generation by preserving frame pointers and debug info.


To build with a specific preset:
```
cmake --preset <preset_name>
cmake --build --preset <preset_name>
```

### Build Preset Reference

| Preset             | `-O3` | `-g` | `-G` (CUDA) | `-pg` | `-fno-omit-frame-pointer` | 🧠 Use Case                         |
|--------------------|:-----:|:----:|:-----------:|:-----:|:--------------------------:|------------------------------------|
| 🟢 `debug`          | ❌    | ✅   | ✅          | ❌    | ✅                          | Full source-level debugging (host + CUDA) |
| 🔴 `release`        | ✅    | ❌   | ❌          | ❌    | ❌                          | Optimized build with fast math      |
| 🟡 `release-profiling` | ✅ | ❌   | ❌          | ✅    | ❌                          | Instrumented build for `gprof`      |
| 🟠 `perf-profiling` | ✅    | ✅   | ❌          | ❌    | ✅                          | Linux `perf`, flamegraphs           |

Legend:
- ✅ = Enabled
- ❌ = Disabled
- `-G`: CUDA device code debugging
- `-pg`: Profiling with `gprof`
- `-fno-omit-frame-pointer`: Required for good stack traces in `perf`



## Debugging & Profiling

The project's Dockerfiles and `docker-compose.yml` are configured to enable GPU debugging and profiling tools inside the container:

- GPU access is enabled via the NVIDIA Container Toolkit.

- Necessary permissions (SYS_PTRACE, seccomp=unconfined) and shared memory settings (/tmp, /dev/shm) are preconfigured for debugger compatibility.

### ✅ In-Container Tools
Once inside the container (e.g., via docker exec -it hpc_mpich /bin/zsh), you can:

- Use `cuda-gdb` directly from the command line to debug CUDA kernels.

- Access NVIDIA Nsight Compute CLI (ncu) to collect detailed kernel performance metrics.

- Integrate with the VS Code NVIDIA Nsight Extension, after connecting VS Code to the running container using the Dev Containers extension.

> [!NOTE]
> While the VS Code Nsight extension can detect your container and attach to your binary, functionality in the integrated debug terminal is limited (most views show as "unavailable" at this time).


### 🚧 Full Nsight Compute GUI
To use the full Nsight Compute GUI or Nsight Systems, including timeline views and interactive charts, you will likely need to run the application natively (bare metal). These tools currently have limited support inside Docker containers.

## CUDA Design Strategy

The CUDA implementation in this project is optimized to take full advantage of shared memory within each Streaming Multiprocessor (SM). The core idea is to assign large thread blocks to each SM to maximize intra-block parallelism and reduce reliance on slower global memory.

- Each tile of the matrix is computed entirely within a thread block.
- Shared memory is used to store intermediate prefix sum results, allowing fast intra-block communication.
- Global memory is used only for final outputs and offsets between tiles.

This approach reduces global memory traffic and helps achieve high performance on large input arrays. The kernel structure is designed to minimize synchronization costs and maximize instruction-level parallelism on modern NVIDIA GPUs.

## Potential Enhancements
This project is functional and performs well on both MPI and CUDA backends, but several enhancements could be explored in the future to further improve performance and scalability:

- **Tiled Asynchronous Data Transfers (CUDA)**  
Currently, the entire input array is copied from host to device memory before any computation begins. This can result in noticeable overhead for large arrays. A potential improvement would be to load and process data in a tile-wise fashion using CUDA streams. By overlapping memory transfers and computation (e.g., using cudaMemcpyAsync and multiple streams), the pipeline could reduce idle time and increase overall throughput.

- **Multi-GPU Support**  
For very large arrays, partitioning the workload across multiple GPUs could further reduce wall time. This would require extending the current logic to coordinate data distribution and aggregation between devices.

- **Hybrid MPI + CUDA Execution**
While the current implementation supports either an MPI or CUDA backend, integrating both in a hybrid mode could allow for larger-scale deployment across distributed multi-GPU clusters.

- **Adaptive Chunk Size and Block Size**
Current chunk/block sizes are set statically. A profiling-based mechanism could tune these dynamically based on input array dimensions or hardware characteristics.

- **Benchmarking and Visualization Tools**
Adding more detailed benchmarking and visual output for prefix sum correctness and performance profiles (e.g., via Nsight or nvprof) could aid analysis and showcase performance gains clearly.

- **Error Handling and Robustness**
While basic CUDA error checks are included, deeper integration with tools like cuda-memcheck, guard patterns, and extended diagnostics could harden the kernel code.

- **Memory Alignment in CUDA Kernels**: In the final steps of CUDA computation (during offset adjustments after copying tile results to global memory), we perform column-wise prefix sums on a buffer array. This buffer is currently stored in row-major order, which can lead to inefficient memory access when traversing column-wise. Switching to column-major storage or using pitched memory allocations could improve cache coherence and memory throughput for this step.

- **Dynamic Tiling Strategy**: Currently, tile sizes are fixed at runtime. Adding logic to dynamically determine optimal tile dimensions based on input size and GPU architecture could further improve performance.

- **Hybrid Backends**: For mid-sized arrays, a hybrid approach using both MPI and CUDA (e.g., CUDA per node + MPI between nodes) could deliver even better scalability.

- **Nsight Systems Tracing**: Using Nsight Systems to trace kernel execution and memory transfer patterns may reveal additional optimization opportunities.












