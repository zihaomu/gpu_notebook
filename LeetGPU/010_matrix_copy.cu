#include <cuda_runtime.h>
/*
Matrix Copy
Easy
Implement a program that copies an 
 matrix of 32-bit floating point numbers from input array 
 to output array 
 on the GPU. The program should perform a direct element-wise copy so that 
 for all valid indices.

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in matrix B
Example 1:
Input:  A = [[1.0, 2.0],
             [3.0, 4.0]]
Output: B = [[1.0, 2.0],
             [3.0, 4.0]]
Example 2:
Input:  A = [[5.5, 6.6, 7.7],
             [8.8, 9.9, 10.1],
             [11.2, 12.3, 13.4]]
Output: B = [[5.5, 6.6, 7.7],
             [8.8, 9.9, 10.1],
             [11.2, 12.3, 13.4]]
Constraints
1 ≤ N ≤ 4096
All elements are 32-bit floating point numbers
*/
/*
主要考怎么 快速拷贝数据？
一个thread 只拷贝一个数据是否可以？
*/
__global__ void copy_matrix_kernel(const float* A, float* B, int N) {

    size_t index = blockIdx.x * 256 + threadIdx.x;

    if (index < N*N)
    {
        B[index] = A[index];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    cudaDeviceSynchronize();
} 