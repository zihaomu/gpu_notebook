import torch
import time
import torch.nn.functional as F
import triton
import triton.language as tl

def functional_softmax_torch(x):
    return F.softmax(x, dim=-1)

def softmax_safe(x):
    # 减去每行最大值
    x_max = torch.max(x, dim=-1, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / torch.sum(x_exp, dim=-1, keepdim=True)

def softmax_online(x):
    B, N = x.shape
    max_val = torch.full((B,), float('-inf'), device=x.device, dtype=x.dtype)
    sum_exp = torch.zeros((B,), device=x.device, dtype=x.dtype)

    for i in range(N):
        current = x[:, i]
        old_max = max_val
        new_max = torch.maximum(old_max, current)
        # ✅ 注意：更新时要考虑 old 和 new 的相对变化
        sum_exp = sum_exp * torch.exp(old_max - new_max) + torch.exp(current - new_max)
        max_val = new_max

    # 最终 softmax 归一化
    return torch.exp(x - max_val[:, None]) / sum_exp[:, None]

@triton.jit
def softmax_online_kernel(
    X_ptr, Y_ptr,
    B, N,
    stride_xb, stride_xn,
    stride_yb, stride_yn,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    b = pid  # batch index

    # base pointers for batch b
    x_ptrs = X_ptr + b * stride_xb # 跳转到对应的行
    y_ptrs = Y_ptr + b * stride_yb # 跳转到对应的行

    # initialize max and sum_exp
    max_val = tl.full((1,), -float('inf'), tl.float32)
    sum_exp = tl.zeros((1,), tl.float32)

    # iterate over chunks
    for offset in range(0, N, BLOCK_SIZE): # 一次处理一个block的值
        # 一个block内部会分配BLOCK_SIZE个线程，一次处理BLOCK_SIZE个元素
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(x_ptrs + idx * stride_xn, mask=mask, other=-float('inf')) # 加载数据

        # compute online update
        old_max = max_val
        new_max = tl.maximum(old_max, tl.max(x, axis=0)) # 计算新的max

        # update sum_exp
        sum_exp = sum_exp * tl.exp(old_max - new_max) + tl.sum(tl.exp(x - new_max), axis=0)
        max_val = new_max

    # 第二遍归一化
    for offset in range(0, N, BLOCK_SIZE):
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(x_ptrs + idx * stride_xn, mask=mask, other=-float('inf'))
        y = tl.exp(x - max_val) / sum_exp
        tl.store(y_ptrs + idx * stride_yn, y, mask=mask)


def softmax_online_triton(x: torch.Tensor, block_size=128):
    assert x.dim() == 2
    B, N = x.shape
    y = torch.empty_like(x)
    grid = (B,) # 一行一个kernel
    softmax_online_kernel[grid](
        x, y,
        B, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_SIZE=block_size
    )
    return y

# 使用lse类似safe softmax，是等价的，也能够很好的避免溢出
def softmax_online_lse(x):
    B, N = x.shape
    max_val = torch.full((B,), float('-inf'), device=x.device, dtype=x.dtype)
    lse = torch.zeros((B,), device=x.device, dtype=x.dtype)

    for i in range(N):
        current = x[:, i]
        old_max = max_val
        new_max = torch.maximum(old_max, current)
        s = torch.exp(lse - old_max) if i > 0 else torch.zeros_like(lse)
        lse = new_max + torch.log(s * torch.exp(old_max - new_max) + torch.exp(current - new_max))
        max_val = new_max
    return torch.exp(x - lse[:, None])

# lse 2 version: 减少max_val 的寄存器的使用，在实际的attention 计算中，max_val和lse都需要一个MxN的矩阵存储，减少一个矩阵的存储，就能减少寄存器的压力。
def softmax_online_lse_v2(x):
    B, N = x.shape
    lse = torch.zeros((B,), device=x.device, dtype=x.dtype)

    for i in range(N):
        current = x[:, i]
        new_max = torch.maximum(lse, current) if i > 0 else current      # 第一次迭代时，lse还没有更新过，直接用current
        s = torch.exp(lse - new_max) if i > 0 else torch.zeros_like(lse) # 这里，第一个元素时，sum_exp代表之前的求和，为0
        lse = new_max + torch.log(s * torch.exp(new_max - lse) + torch.exp(current - new_max))
    return torch.exp(x - lse[:, None])

if __name__ == "__main__":
    # Example usage
    m = 1024
    n = 1024
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_type = torch.float32
    is_causal = False
    
    warmup_runs = 2
    repeat_time = 10
    
    function_to_test = softmax_online_lse_v2
    reference_function = functional_softmax_torch

    input_tensor = torch.randn((m, n), device=device, dtype=data_type)

    for _ in range(warmup_runs):
        # Warm-up
        function_to_test(input_tensor)

    time_starts = time.time()
    for _ in range(repeat_time):
        reference_function(input_tensor)
        torch.cuda.synchronize() if device == 'cuda' else None
    
    elapsed = (time.time() - time_starts) / (repeat_time * 1.0)
    print(f"Reference Average time per run: {elapsed * 1000:.3f} ms")

    time_starts = time.time()
    for _ in range(repeat_time):
        output = function_to_test(input_tensor)
        torch.cuda.synchronize() if device == 'cuda' else None
    
    elapsed = (time.time() - time_starts) / (repeat_time * 1.0)
    print(f"Test Average time per run: {elapsed * 1000:.3f} ms")

    # Validate the output
    reference_output = reference_function(input_tensor)
    # print first 5 elements of both outputs for comparison
    print("Output (first 5 elements):", output.flatten()[:5])
    print("Reference (first 5 elements):", reference_output.flatten()[:5])
    
    # compute cosine similarity
    cosine_similarity = torch.nn.functional.cosine_similarity(output.flatten(), reference_output.flatten(), dim=0)
    print("Cosine similarity:", cosine_similarity.item())

    