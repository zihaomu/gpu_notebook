# triton_gemm_occupancy_examples.py
# Three Triton GEMM experiments to demonstrate occupancy effects:
# 1) reg_heavy_gemm: force high register pressure by using many accumulators / large tiles
# 2) smem_heavy_gemm: force high shared-memory (conceptual) pressure by using large block tiling
# 3) balanced_gemm: a tuned kernel with moderate registers and small shared usage
#
# NOTE:
# - This file is a runnable starting point. Triton API changes over time; adapt constants if needed.
# - Measure with `nsys profile -o trace python run_examples.py` or use `nvprof`/`nv-nsight`.
# - Expect: reg_heavy -> low occupancy due to registers; smem_heavy -> low occupancy due to per-block SMEM; balanced -> high throughput.

import torch
import triton
import triton.language as tl
import time

# Utility: launcher for Triton kernels (matrix multiply C = A @ B)

def rand_mat(M, N, device='cuda', dtype=torch.float32):
    return torch.randn((M, N), device=device, dtype=dtype)


# ---------------------------
# 1) Register-heavy GEMM
# ---------------------------
# Strategy: use a large BLOCK_M x BLOCK_N tile and an explicit inner loop unrolled into many
# local accumulator variables. The many accumulators per program increase per-thread register
# demand. On GPUs with finite registers per SM this reduces the number of concurrent blocks.

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

    # initialize many accumulators to force registers
    # We'll create BLOCK_M x BLOCK_N accumulators explicitly by tiling small (for demo)
    # For readability we use a tensor of accumulators (this still maps to registers)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # load A tile and B tile
        a = tl.load(A_ptr + (offs_m[:, None] * stride_am + (k + tl.arange(0, BLOCK_K))[None, :] * stride_ak))
        b = tl.load(B_ptr + ((k + tl.arange(0, BLOCK_K))[:, None] * stride_bk + offs_n[None, :] * stride_bn))
        # manual multiply-accumulate (unrolled by BLOCK_K at the language level)
        # this inner loop will produce many temporary values and keep many values live
        for kk in range(BLOCK_K):
            # accumulate - this line will create temporaries per kk
            acc += a[:, kk:kk+1] * b[kk:kk+1, :]
    # store
    tl.store(C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn), acc)


# ---------------------------
# 2) Shared-memory-heavy GEMM (conceptual)
# ---------------------------
# Strategy: Use very large BLOCK_K and BLOCK_M/N so each block would ideally cache
# substantial parts of A/B in shared memory. In Triton we emulate this by using
# big tiles and copying whole tiles inside the block (which in native CUDA would use
# shared memory). This causes large per-block shared-memory usage and reduces the
# number of blocks that can be resident per SM.

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

    # "scratch" tile: conceptually this would be shared memory in CUDA
    # We allocate a local array that Triton will try to map efficiently; large sizes
    # increase per-block shared memory demand.
    A_tile = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    B_tile = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # copy into tile (emulate shared loads)
        A_tile = tl.load(A_ptr + (offs_m[:, None] * stride_am + (k + tl.arange(0, BLOCK_K))[None, :] * stride_ak))
        B_tile = tl.load(B_ptr + ((k + tl.arange(0, BLOCK_K))[:, None] * stride_bk + offs_n[None, :] * stride_bn))
        # compute using the tiles
        for kk in range(BLOCK_K):
            acc += A_tile[:, kk:kk+1] * B_tile[kk:kk+1, :]
    tl.store(C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn), acc)


# ---------------------------
# 3) Balanced GEMM
# ---------------------------
# Strategy: use moderate tile sizes, rely on tl.dot (vectorized) and keep working set small
# so neither registers nor per-block scratch explode.

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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptr + (offs_m[:, None] * stride_am + (k + tl.arange(0, BLOCK_K))[None, :] * stride_ak))
        b = tl.load(B_ptr + ((k + tl.arange(0, BLOCK_K))[:, None] * stride_bk + offs_n[None, :] * stride_bn))
        # use vectorized dot to keep code compact and encourage efficient codegen
        acc += tl.dot(a, b)
    tl.store(C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn), acc)


# ---------------------------
# Runner: create matrices and launch kernels with different tile sizes
# ---------------------------

def run_experiment(kernel, M=4096, N=4096, K=4096, dtype=torch.float32,
                   BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, iters=10):
    A = rand_mat(M, K, dtype=dtype)
    B = rand_mat(K, N, dtype=dtype)
    C = torch.empty((M, N), device='cuda', dtype=dtype)

    # prepare strides / pointers
    A_ptr = A.data_ptr()
    B_ptr = B.data_ptr()
    C_ptr = C.data_ptr()

    # launch grid
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # warmup
    kernel[grid](A_ptr, B_ptr, C_ptr,
                 M, N, K,
                 A.stride(0), A.stride(1),
                 B.stride(0), B.stride(1),
                 C.stride(0), C.stride(1),
                 BLOCK_M, BLOCK_N, BLOCK_K)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        kernel[grid](A_ptr, B_ptr, C_ptr,
                     M, N, K,
                     A.stride(0), A.stride(1),
                     B.stride(0), B.stride(1),
                     C.stride(0), C.stride(1),
                     BLOCK_M, BLOCK_N, BLOCK_K)
    torch.cuda.synchronize()
    t1 = time.time()
    avg_ms = (t1 - t0) * 1000.0 / iters
    gflops = 2.0 * M * N * K / (avg_ms * 1e-3) / 1e9
    print(f"{kernel.__name__}: avg {avg_ms:.3f} ms, {gflops:.1f} GFLOP/s (BLOCK {BLOCK_M}x{BLOCK_N}x{BLOCK_K})")
    return avg_ms, gflops


if __name__ == "__main__":
    # Example runs (adjust BLOCK sizes to provoke different resource pressure)
    # 1) Reg-heavy: large BLOCK_M/BLOCK_N and moderate BLOCK_K
    run_experiment(reg_heavy_matmul, M=2048, N=2048, K=2048, BLOCK_M=128, BLOCK_N=128, BLOCK_K=16)

    # 2) SMEM-heavy: very large BLOCK_K so per-block tile is big
    run_experiment(smem_heavy_matmul, M=2048, N=2048, K=2048, BLOCK_M=64, BLOCK_N=64, BLOCK_K=256)

    # 3) Balanced: moderate tiles
    run_experiment(balanced_matmul, M=2048, N=2048, K=2048, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)

    print("Done")
