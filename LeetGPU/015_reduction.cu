
/*
Reduction
Medium
Write a GPU program that performs parallel reduction on an array of 32-bit floating point numbers to compute their sum. The program should take an input array and produce a single output value containing the sum of all elements.

Implementation Requirements
Use only GPU native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output variable
Example 1:
Input: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
Output: 36.0
Example 2:
Input: [-2.5, 1.5, -1.0, 2.0]
Output: 0.0
Constraints
1 ≤ N ≤ 100,000,000
-1000.0 ≤ input[i] ≤ 1000.0
The final sum will always fit within a 32-bit float
*/
// input, output are device pointers

/*
问题分析：

*/
#include <cuda_runtime.h>

__global__ void sum_kernel(const float* input, float* output, int N, int N_4) {
    __shared__ float sdata[256];
    size_t index = blockIdx.x * 256 + threadIdx.x;
    int tid = threadIdx.x;
    volatile float sum = 0.f;

    if (index < N_4) {
        size_t base_index = index * 4;
        if (base_index < N) sum += input[base_index];
        if (base_index + 1 < N) sum += input[base_index + 1];
        if (base_index + 2 < N) sum += input[base_index + 2];
        if (base_index + 3 < N) sum += input[base_index + 3];
    }

    sdata[tid] = float(sum);
    __syncthreads();

    // 归约求和
    for (unsigned int s = 256 / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 使用共享内存的结果
    if (tid == 0) atomicAdd(output, sdata[0]);
}

extern "C" void solve(const float* input, float* output, int N) {  
    int threadsPerBlock = 256;
    int N_4 = (N + 3) / 4;
    int blocksPerGrid = (N_4 + threadsPerBlock - 1) / threadsPerBlock;
    
    float h_zero = 0.0f;
    cudaMemcpy(output, &h_zero, sizeof(float), cudaMemcpyHostToDevice);
    
    sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, N_4);
    cudaDeviceSynchronize();
}
