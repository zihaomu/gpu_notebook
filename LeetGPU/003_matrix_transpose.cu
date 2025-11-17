#include <cuda_runtime.h>

/*
一个kernel 翻转 一个 16x16的方块，确定越界处理。
*/
#define TILE_DIM 16
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {

    // input index
    size_t row_i = blockIdx.y * TILE_DIM + threadIdx.y;
    size_t col_i = blockIdx.x * TILE_DIM + threadIdx.x;

    size_t row_o = col_i;
    size_t col_o = row_i;

    __shared__ float tile[TILE_DIM][TILE_DIM];

    // load data into shared memory
    if (row_i < rows && col_i < cols)
        tile[threadIdx.y][threadIdx.x] = input[row_i * cols + col_i];
    __syncthreads();

    // write back
    if (row_i < rows && col_i < cols) // 为什么可以使用同样条件判断越界？ 因为row_i 是等于col_o的，等价的
        output[row_o * rows + col_o] = tile[threadIdx.y][threadIdx.x]; // 同时要注意，这里的row_o 中包含threadIdx.x，而维度上应该对应threadIdx.y，需要交错，才能达到transpose的效果。
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}