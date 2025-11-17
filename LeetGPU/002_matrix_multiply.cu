#include <cuda_runtime.h>


__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) 
{
    /*
    并行的难点之一，在于 ABC三个矩阵，x y thread和block， 太多参数量，要找个基准，其他的都是和这个基准做对应。
    这里没找对，后面很多计算都容易错误。

    这里，以C为基准，blockIdx.y 和 threadIdx.y 对应 row
    blockIdx.x 和 threadIdx.x 对应 col。
    在加载A时，threadIdx.y对应A的row，threadIdx.x对应A的col
    在加载B时，threadIdx.y对应B的row，threadIdx.x对应B的col

    这样就能统一起来。

    当然，这样计算不是最快的，因为有 bank冲突的问题，不过先实现正确再说。
    */
    size_t row = blockIdx.y * 16 + threadIdx.y;
    size_t col = blockIdx.x * 16 + threadIdx.x;

    __shared__ float a_tile[16][16];
    __shared__ float b_tile[16][16];
    float v = 0.0f;

    for (int i = 0; i < N; i += 16)
    {
        /*
        记录一下一下午的报错：
        在这里提前 把row乘上了N，之后导致
        a_row = row * N，在后面计算 a_row < M 时，就会导致错误分支，导致一直报错。
        */

        //load a 
        size_t a_row = row;
        size_t a_col = i + threadIdx.x;
        a_tile[threadIdx.y][threadIdx.x] = (a_row < M && a_col < N) ? A[a_row * N + a_col] : 0.f;
        //load b
        size_t b_row = (i + threadIdx.y);
        size_t b_col = col;
        b_tile[threadIdx.y][threadIdx.x] = (b_row < N && b_col < K) ? B[b_row * K + b_col] : 0.f;

        __syncthreads();
        
        for (int k = 0; k < 16; k++)
        {
            v += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < K)
    {
        *(C + row * K + col) = v;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x, // 切成16x16来计算
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}