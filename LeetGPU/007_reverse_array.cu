#include <cuda_runtime.h>
/*

Reverse Array
Easy
Implement a program that reverses an array of 32-bit floating point numbers in-place. The program should perform an in-place reversal of input.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored back in input
Example 1:
Input: [1.0, 2.0, 3.0, 4.0]
Output: [4.0, 3.0, 2.0, 1.0]
Example 2:
Input: [1.5, 2.5, 3.5]
Output: [3.5, 2.5, 1.5]
Constraints
1 ≤ N ≤ 100,000,000
*/

/*
如何将任务进行划分？
将前后分成两半，
也就是只有一半任务需要处理。
*/

__global__ void reverse_array(float* input, int N, int N_2) 
{
    size_t index = blockIdx.x * 256 + threadIdx.x;

    if (index < N_2)
    {
        // swap input
        float v = input[index];
        input[index] = input[N - index - 1]; // 注意：这个下标必须是 N-index-1而不能是N-index
        input[N - index - 1] = v;
    }
}
// input is device pointer
extern "C" void solve(float* input, int N) {
    int N_2 = N / 2;
    int threadsPerBlock = 256;

    int blocksPerGrid = (N_2 + threadsPerBlock - 1) / (threadsPerBlock);

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N, N_2);
    cudaDeviceSynchronize();
}