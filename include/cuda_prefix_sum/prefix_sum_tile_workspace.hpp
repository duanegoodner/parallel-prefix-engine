#pragma once

#include <cuda_runtime.h>



struct PrefixSumTileWorkspace {
    int* d_right_edges = nullptr;
    int* d_bottom_edges = nullptr;
  
    void Allocate(int array_height, int array_width, int tile_height, int tile_width) {
      int tiles_x = (array_height + tile_height - 1) / tile_height;
      int tiles_y = (array_width + tile_width - 1) / tile_width;
  
      cudaMalloc(&d_right_edges, array_height * (tiles_y - 1) * sizeof(int));
      cudaMalloc(&d_bottom_edges, (tiles_x - 1) * array_width * sizeof(int));
    }
  
    void Free() {
      cudaFree(d_right_edges);
      cudaFree(d_bottom_edges);
    }
  };
  