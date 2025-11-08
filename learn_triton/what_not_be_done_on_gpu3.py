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
# 2) 极端的寄存器压力版本 - 手动展开循环
# ---------------------------
@triton.jit
def extreme_reg_pressure_matmul(A_ptr, B_ptr, C_ptr,
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

    # 手动创建多个独立的累加器 - 这是增加寄存器压力的最直接方法
    NUM_ACCUMULATORS = 16
    # 我们必须手动创建每个累加器，因为Triton不支持动态张量索引
    
    # 累加器组0
    acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # 累加器组1
    acc4 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc5 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc6 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc7 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # 累加器组2
    acc8 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc9 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # 累加器组3
    acc12 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc13 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc14 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc15 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    acc16 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc17 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc18 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc19 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    acc20 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc21 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc22 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc23 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak
        b_ptrs = B_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        
        dot_result = tl.dot(a, b)
        
        # 手动更新每个累加器 - 这会创建很多活跃的寄存器
        weight = 1.0 / NUM_ACCUMULATORS
        
        # 组0
        acc0 += dot_result * (weight * 1.00)
        acc1 += dot_result * (weight * 0.99)
        acc2 += dot_result * (weight * 0.98)
        acc3 += dot_result * (weight * 0.97)
        
        # 组1
        acc4 += dot_result * (weight * 0.96)
        acc5 += dot_result * (weight * 0.95)
        acc6 += dot_result * (weight * 0.94)
        acc7 += dot_result * (weight * 0.93)
        
        # 组2
        acc8 += dot_result * (weight * 0.92)
        acc9 += dot_result * (weight * 0.91)
        acc10 += dot_result * (weight * 0.90)
        acc11 += dot_result * (weight * 0.89)
        
        # 组3
        # acc12 += dot_result * (weight * 0.88)
        acc13 += dot_result * (weight * 0.87)
        acc14 += dot_result * (weight * 0.86)
        acc15 += dot_result * (weight * 0.85)

        acc16 += dot_result * (weight * 0.84)
        acc17 += dot_result * (weight * 0.83)
        acc18 += dot_result * (weight * 0.82)
        acc19 += dot_result * (weight * 0.81)
        
        acc20 += dot_result * (weight * 0.80)
        acc21 += dot_result * (weight * 0.79)
        acc22 += dot_result * (weight * 0.78)
        acc23 += dot_result * (weight * 0.77)

    # 手动合并所有累加器
    result = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    result += acc0 + acc1 + acc2 + acc3
    result += acc4 + acc5 + acc6 + acc7
    result += acc8 + acc9 + acc10 + acc11
    result += acc12 + acc13 + acc14 + acc15
    result += acc16 + acc17 + acc18 + acc19
    result += acc20 + acc21 + acc22 + acc23
    
    # Store result
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, result)

# ---------------------------
# 3) 共享内存压力版本 - 使用更小的块尺寸
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
        # 使用合理的块尺寸
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
# 4) Balanced GEMM - 使用适中的块尺寸
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
# 5) 极端的共享内存压力版本 - 使用更小的块但更多阶段
# ---------------------------
@triton.jit
def extreme_smem_pressure_matmul(A_ptr, B_ptr, C_ptr,
                                M, N, K,
                                stride_am, stride_ak,
                                stride_bk, stride_bn,
                                stride_cm, stride_cn,
                                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                NUM_STAGES: tl.constexpr = 4):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 使用多阶段来增加共享内存压力
    for k in range(0, K, BLOCK_K):
        # 初始化阶段累加器
        stage_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # 多次加载相同数据来模拟高共享内存使用
        for stage in range(NUM_STAGES):
            a_ptrs = A_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak
            b_ptrs = B_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn
            
            a_tile = tl.load(a_ptrs)
            b_tile = tl.load(b_ptrs)
            
            # 每个阶段都进行计算但只累加一次
            if stage == 0:
                stage_acc = tl.dot(a_tile, b_tile)
            else:
                stage_acc += tl.dot(a_tile, b_tile) * 0.1  # 小权重避免数值问题
        
        acc += stage_acc / NUM_STAGES
 
    # Store result
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, acc)


# ---------------------------
# Runner function
# ---------------------------
def run_experiment(kernel, kernel_name, M=2048, N=2048, K=2048, dtype=torch.float32,
                   BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, iters=1, **kwargs):
    try:
        A = rand_mat(M, K, dtype=dtype)
        B = rand_mat(K, N, dtype=dtype)
        C = torch.empty((M, N), device='cuda', dtype=dtype)

        # launch grid
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        print(f"Launching {kernel_name} with grid {grid}, blocks {BLOCK_M}x{BLOCK_N}x{BLOCK_K}")

        # warmup
        if kwargs:
            kernel[grid](A, B, C, M, N, K,
                        A.stride(0), A.stride(1),
                        B.stride(0), B.stride(1),
                        C.stride(0), C.stride(1),
                        BLOCK_M, BLOCK_N, BLOCK_K, **kwargs)
        else:
            kernel[grid](A, B, C, M, N, K,
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
            if kwargs:
                kernel[grid](A, B, C, M, N, K,
                            A.stride(0), A.stride(1),
                            B.stride(0), B.stride(1),
                            C.stride(0), C.stride(1),
                            BLOCK_M, BLOCK_N, BLOCK_K, **kwargs)
            else:
                kernel[grid](A, B, C, M, N, K,
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
    
    # 配置参数
    configs = [
        # (kernel, name, BLOCK_M, BLOCK_N, BLOCK_K, description, kwargs)
        (extreme_reg_pressure_matmul, "EXTREME_REG", 64, 64, 32, "Manual 24 accumulators", {}),
        (reg_heavy_matmul, "REG_HEAVY", 64, 64, 32, "4 accumulators", {}),
        (extreme_smem_pressure_matmul, "EXTREME_SMEM", 256, 256, 32, "Multi-stage SMEM", {"NUM_STAGES": 4}),
        (smem_heavy_matmul, "SMEM_HEAVY", 64, 64, 32, "Standard", {}),
        (balanced_matmul, "BALANCED", 64, 64, 32, "Moderate tile sizes", {}),
    ]
    
    results = {}
    
    for kernel, name, bm, bn, bk, desc, kwargs in configs:
        print(f"\n{name}: {desc}")
        print("-" * 40)
        time_ms, gflops = run_experiment(kernel, name, 
                                       M=2048, N=2048, K=2048,
                                       BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, 
                                       iters=3, **kwargs)
        if time_ms is not None:
            results[name] = (time_ms, gflops)
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    for name, (time_ms, gflops) in sorted(results.items(), key=lambda x: x[1][1], reverse=True):
        print(f"{name:20} : {time_ms:6.3f} ms, {gflops:8.1f} GFLOP/s")
    
    print("\nAnalysis:")
    print("- High register pressure kernels use more registers (check compiler output)")
    print("- Lower GFLOP/s indicates lower occupancy due to resource constraints")
    print("- Balanced kernel should achieve the best performance")
    print("\nDone!")