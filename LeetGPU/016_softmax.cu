#include <cuda_runtime.h>
/*

Softmax
Medium
Write a program that computes the softmax function for an array of 32-bit floating-point numbers on a GPU. The softmax function is defined as follows:

For an input array 
 of length 
, the softmax of 
, denoted 
, is an array of length 
 where the 
-th element is:

 
 
 

Your solution should handle potential overflow issues by using the "max trick". Subtract the maximum value of the input array from each element before exponentiation.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the array output
Example 1:
Input: [1.0, 2.0, 3.0], N = 3
Output: [0.090, 0.244, 0.665] (approximately)
Example 2:
Input: [-10.0, -5.0, 0.0, 5.0, 10.0], N = 5
Output: [2.04e-09, 4.52e-07, 9.99e-01, 2.26e-02, 9.77e-01] (approximately)
Constraints
1 ≤ N ≤ 500,000
*/

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__inline__ __device__ float warpReduceMax(float val, int warp_size) {
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }

    __syncwarp();
    return val; // 返回当前线程处理的最大值
}

__inline__ __device__ float warpReduceSum(float val, int warp_size) {
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    __syncwarp();
    return val; // 返回当前线程处理的和
}

__global__ void softmax_kernel(const float* input, float* output, int N) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_tx = threadIdx.x - warp_id * WARP_SIZE;

    
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}