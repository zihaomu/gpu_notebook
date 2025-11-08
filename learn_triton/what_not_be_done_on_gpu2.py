# triton_gemm_occupancy_examples.py
# Three Triton GEMM experiments to demonstrate occupancy effects:

import torch
import triton
import triton.language as tl
import time

def rand_mat(M, N, device='cuda', dtype=torch.float32):
    return torch.randn((M, N), device=device, dtype=dtype)


# ---------------------------
# 1) Register-heavy GEMM - 使用多个独立的累加器
# ---------------------------
@triton.jit
def reg_heavy_matmul(A_ptr, B_ptr, C_ptr,
                     M, N, K,
                     stride_am, stride_ak,
                     stride_bk, stride_bn,
                     stride_cm, stride_cn,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # block row/col offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 创建多个独立的累加器来增加寄存器压力
    acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Load A tile
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak
        a = tl.load(a_ptrs)
        
        # Load B tile  
        b_ptrs = B_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs)
        
        # 使用tl.dot但将结果分散到多个累加器中
        # 这样每个累加器都会占用寄存器
        dot_result = tl.dot(a, b)
        acc0 += dot_result * 0.25
        acc1 += dot_result * 0.25
        acc2 += dot_result * 0.25
        acc3 += dot_result * 0.25
    
    # 合并结果
    result = acc0 + acc1 + acc2 + acc3
    
    # Store result
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, result)


# ---------------------------
# 2) Shared-memory-heavy GEMM - 使用大块尺寸
# ---------------------------
@triton.jit
def smem_heavy_matmul(A_ptr, B_ptr, C_ptr,
                      M, N, K,
                      stride_am, stride_ak,
                      stride_bk, stride_bn,
                      stride_cm, stride_cn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # 使用大块尺寸来增加共享内存压力
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak
        a_tile = tl.load(a_ptrs)
        
        b_ptrs = B_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_tile = tl.load(b_ptrs)
        
        # 简单的矩阵乘法
        acc += tl.dot(a_tile, b_tile)
    
    # Store result
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, acc)


# ---------------------------
# 3) Balanced GEMM - 使用适中的块尺寸
# ---------------------------
@triton.jit
def balanced_matmul(A_ptr, B_ptr, C_ptr,
                    M, N, K,
                    stride_am, stride_ak,
                    stride_bk, stride_bn,
                    stride_cm, stride_cn,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Load tiles
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak
        b_ptrs = B_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        
        # Use efficient matrix multiplication
        acc += tl.dot(a, b)
    
    # Store result
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, acc)


# ---------------------------
# # 4) 极端的寄存器压力版本 - 使用更多累加器
# # ---------------------------
# @triton.jit
# def extreme_reg_pressure_matmul(A_ptr, B_ptr, C_ptr,
#                                M, N, K,
#                                stride_am, stride_ak,
#                                stride_bk, stride_bn,
#                                stride_cm, stride_cn,
#                                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
#     pid_m = tl.program_id(0)
#     pid_n = tl.program_id(1)

#     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
#     offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
#     offs_k = tl.arange(0, BLOCK_K)

#     # 创建很多累加器来最大化寄存器压力
#     NUM_ACCUMULATORS = 8
#     accumulators = []
#     for i in range(NUM_ACCUMULATORS):
#         accumulators.append(tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))

#     for k in range(0, K, BLOCK_K):
#         a_ptrs = A_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak
#         b_ptrs = B_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
#         a = tl.load(a_ptrs)
#         b = tl.load(b_ptrs)
        
#         dot_result = tl.dot(a, b)
        
#         # 将结果分散到所有累加器中
#         scale = 1.0 / NUM_ACCUMULATORS
#         for i in range(NUM_ACCUMULATORS):
#             accumulators[i] += dot_result * scale
    
#     # 合并所有累加器
#     result = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
#     for i in range(NUM_ACCUMULATORS):
#         result += accumulators[i]
    
#     # Store result
#     offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
#     offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
#     c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
#     tl.store(c_ptrs, result)


# ---------------------------
# Runner function with better error handling
# ---------------------------

def run_experiment(kernel, kernel_name, M=2048, N=2048, K=2048, dtype=torch.float32,
                   BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, iters=3):
    try:
        A = rand_mat(M, K, dtype=dtype)
        B = rand_mat(K, N, dtype=dtype)
        C = torch.empty((M, N), device='cuda', dtype=dtype)

        # launch grid
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        print(f"Launching {kernel_name} with grid {grid}, blocks {BLOCK_M}x{BLOCK_N}x{BLOCK_K}")

        # warmup
        kernel[grid](A, B, C,
                     M, N, K,
                     A.stride(0), A.stride(1),
                     B.stride(0), B.stride(1),
                     C.stride(0), C.stride(1),
                     BLOCK_M, BLOCK_N, BLOCK_K)
        torch.cuda.synchronize()

        # Time the kernel
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(iters):
            kernel[grid](A, B, C,
                         M, N, K,
                         A.stride(0), A.stride(1),
                         B.stride(0), B.stride(1),
                         C.stride(0), C.stride(1),
                         BLOCK_M, BLOCK_N, BLOCK_K)
        end_event.record()
        torch.cuda.synchronize()
        
        avg_ms = start_event.elapsed_time(end_event) / iters
        
        if avg_ms > 0:
            gflops = 2.0 * M * N * K / (avg_ms * 1e-3) / 1e9
            print(f"✓ {kernel_name}: {avg_ms:.3f} ms, {gflops:.1f} GFLOP/s")
            return avg_ms, gflops
        else:
            print(f"⚠ {kernel_name}: Time measurement too small")
            return 0, 0
            
    except Exception as e:
        print(f"✗ {kernel_name} failed: {str(e)}")
        return None, None


if __name__ == "__main__":
    print("Running Triton GEMM Occupancy Experiments...")
    print("=" * 60)
    
    # Test configurations
    configs = [
        # (kernel, name, BLOCK_M, BLOCK_N, BLOCK_K, description)
        # (extreme_reg_pressure_matmul, "EXTREME_REG_PRESSURE", 128, 128, 32, "Many accumulators + large tiles"),
        (reg_heavy_matmul, "REG_HEAVY", 64, 64, 32, "Multiple accumulators"), # 524288 = 0.5 MB
        (smem_heavy_matmul, "SMEM_HEAVY", 64, 64, 64, "Very large K dimension"), # 262144 = 0.25 MB
        (balanced_matmul, "BALANCED", 64, 64, 32, "Moderate tile sizes"),         # 65536  = 0.0625 MB
    ]
    
    results = {}
    
    for kernel, name, bm, bn, bk, desc in configs:
        print(f"\n{name}: {desc}")
        print("-" * 40)
        time_ms, gflops = run_experiment(kernel, name, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)
        if time_ms is not None:
            results[name] = (time_ms, gflops)
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    for name, (time_ms, gflops) in results.items():
        print(f"{name:20} : {time_ms:6.2f} ms, {gflops:6.1f} GFLOP/s")
    
    print("\nDone! Compare the performance to see occupancy effects.")