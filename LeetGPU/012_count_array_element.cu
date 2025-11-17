#include <cuda_runtime.h>

/*
Count Array Element
Easy
Write a GPU program that counts the number of elements with the integer value k in an array of 32-bit integers. The program should count the number of elements with k in an array. You are given an input array input of length N and integer k.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output variable
Example 1:
Input: [1, 2, 3, 4, 1], k = 1
Output: 2
Example 2:
Input: [5, 10, 5, 2], k = 11
Output: 0
Constraints
1 ≤ N ≤ 100,000,000
1 ≤ input[i], k ≤ 100,000
*/
#include <cuda_runtime.h>

__global__ void count_equal_kernel(const int* input, int* output, int N, int K, int N_4) {

    size_t index = blockIdx.x * 256 + threadIdx.x;

    int sum = 0;
    if (index * 4 < N && input[index * 4] == K) sum++;
    if (index * 4 + 1 < N && input[index * 4 + 1] == K) sum++;
    if (index * 4 + 2< N && input[index * 4 + 2] == K) sum++;
    if (index * 4 + 3< N && input[index * 4 + 3] == K) sum++;

    if(sum > 0)
    {
        atomicAdd(output, sum);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 256;
    int N_4 = N / 4 + 1;
    int blocksPerGrid = (N_4 + threadsPerBlock - 1) / threadsPerBlock;
    cudaMemset(output, 0, sizeof(int));
    count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K, N_4);
    cudaDeviceSynchronize();
}