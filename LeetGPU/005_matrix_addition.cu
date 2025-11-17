#include <cuda_runtime.h>
/*
Matrix Addition
Easy
Implement a program that performs element-wise addition of two 
 matrices containing 32-bit floating point numbers on a GPU. The program should take two input matrices of equal dimensions and produce a single output matrix containing their element-wise sum.

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in matrix C
Example 1:
Input:  A = [[1.0, 2.0],
             [3.0, 4.0]]
        B = [[5.0, 6.0],
             [7.0, 8.0]]
Output: C = [[6.0, 8.0],
             [10.0, 12.0]]
Example 2:
Input:  A = [[1.5, 2.5, 3.5],
             [4.5, 5.5, 6.5],
             [7.5, 8.5, 9.5]]
        B = [[0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5]]
Output: C = [[2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0],
             [8.0, 9.0, 10.0]]
Constraints
Input matrices A and B have identical dimensions
1 ≤ N ≤ 4096
All elements are 32-bit floating point numbers
*/
__global__ void matrix_add(const float* A, const float* B, float* C, int N) {

    size_t index = blockIdx.x * 256 + threadIdx.x;

    if (index < N * N)
    {
        C[index] = A[index] + B[index];
    }

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
