/*1D Convolution
Easy
Implement a program that performs a 1D convolution operation. Given an input array and a kernel (filter), compute the convolved output. The convolution should be performed with a "valid" boundary condition, meaning the kernel is only applied where it fully overlaps with the input.

The input consists of two arrays:

input: A 1D array of 32-bit floating-point numbers.
kernel: A 1D array of 32-bit floating-point numbers representing the convolution kernel.
The output should be written to the output array, which will have a size of input_size - kernel_size + 1.
The convolution operation is defined mathematically as:

where 
 ranges from 0 to 
.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the array output
Example 1:
Input: input = [1, 2, 3, 4, 5], kernel = [1, 0, -1]
Output: [-2, -2, -2]
Example 2:
Input: input = [2, 4, 6, 8], kernel = [0.5, 0.2]
Output: [1.8, 3.2, 4.6]
Constraints
1 ≤ input_size ≤ 1,500,000
1 ≤ kernel_size ≤ 2047
kernel_size ≤ input_size*/

#include <cuda_runtime.h>

/*
问题分析，
如果从输入 index 推出 输出 index， 则有前后依赖，
而从输出 推输入则不会。可见， 大多数计算都是从输出推输入。

half_kernel=kernel_size / 2, 要处理 当kernel size > input size 的情况。
output_size = input_size - kernl_size + 1
每个输出对应的输入： o_i = kernel @ i_i
其中i_i 表示 对应的数值，中间坐标为 o_i + half_kernel, 范围 i_i - half_kernel，i_i + half_kernel

*/
__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) 
{
    size_t index = blockIdx.x * 256 + threadIdx.x;
    float v = 0;

    if (index < output_size)
    {
        size_t input_index = index;
        for (int i = 0; i < kernel_size)
        {
            v += kernel_size[i] * input[index + i];
        }
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int half_kernel=kernel_size / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}