import numpy as np
import time
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

def functional_scaled_dot_product_attention_naive(query, key, value, attn_mask=None, scale=None, is_causal=False):
    """
    Manual implementation of scaled dot-product attention similar to PyTorch's F.scaled_dot_product_attention.

    Args:
    query: (batch, num_heads, seq_len_q, head_dim)
        key:   (batch, num_heads, seq_len_k, head_dim)
        value: (batch, num_heads, seq_len_k, head_dim)
        attn_mask: optional attention mask, broadcastable to (batch, num_heads, seq_len_q, seq_len_k)
        scale: custom scaling factor. If None, use 1/sqrt(head_dim)
        is_causal: whether to apply causal mask

    Returns:
        attn_output: (batch, num_heads, seq_len_q, head_dim)
    """

    B, H, Lq, D = query.shape
    _, _, Lk, _ = key.shape

    # print q, k, v shapes
    # print(f"Query shape: {query.shape}, Key shape: {key.shape}, Value shape: {value.shape}")

    if scale is None:
        scale = 1.0 / (D ** 0.5)

    # Step 1: Compute raw attention scores
    scores = np.matmul(query, key.transpose(0, 1, 3, 2)) * scale  # shape: (B, H, Lq, Lk)

    # Step 2: Apply masks
    if attn_mask is not None:
        scores = np.where(attn_mask == 0, float('-inf'), scores)

    if is_causal:
        causal_mask = np.tril(np.ones((Lq, Lk), dtype=bool))
        scores = np.where(~causal_mask, float('-inf'), scores)

    # Step 3: Softmax over last dim (Lk)
    attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)

    # Step 4: Compute output
    # value shape: (B, H, Lk, D)
    attn_output = np.matmul(attn_weights, value)  # shape: (B, H, Lq, D)

    return attn_output


def functional_attention_torch(query, key, value, attn_mask=None, scale=None, is_causal=False):
    """
    Manual PyTorch implementation of scaled dot-product attention.

    Args:
        query: (B, H, Lq, D)
        key:   (B, H, Lk, D)
        value: (B, H, Lk, D)
        attn_mask: optional mask broadcastable to (B, H, Lq, Lk),
                   where masked positions are 0 (False) and unmasked are 1 (True)
        scale: optional float, defaults to 1/sqrt(D)
        is_causal: whether to apply causal mask

    Returns:
        attn_output: (B, H, Lq, D)
    """
    B, H, Lq, D = query.shape
    _, _, Lk, _ = key.shape

    if scale is None:
        scale = 1.0 / (D ** 0.5)

    # ---- Step 1: Compute scaled attention scores ----
    # (B, H, Lq, Lk)
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # ---- Step 2: Apply masks ----
    if attn_mask is not None:
        # make sure mask is broadcastable and bool
        mask = attn_mask.to(torch.bool)
        scores = scores.masked_fill(~mask, float('-inf'))

    if is_causal:
        # create lower-triangular mask
        causal_mask = torch.tril(torch.ones(Lq, Lk, device=scores.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float('-inf'))

    # ---- Step 3: Softmax ----
    attn_weights = torch.softmax(scores, dim=-1)

    # ---- Step 4: Weighted sum of values ----
    attn_output = torch.matmul(attn_weights, value)
    torch.cuda.synchronize()
    return attn_output


BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32

# def _triton_attention_kernel()
#     Q_ptr, K_ptr, V_ptr, Out_ptr,

# 这里使用的维度是 (B, H, L, D)，和pytorch 的维度是一样的
# 官方使用的维度是： (B, L, H, D)，需要注意区分：https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py
def functional_attention_triton_v1(query, key, value, attn_mask=None, scale=None, is_causal=False):
    """
    Manual implementation of scaled dot-product attention similar to PyTorch's F.scaled_dot_product_attention.

    Args:
    query: (batch, num_heads, seq_len_q, head_dim)
        key:   (batch, num_heads, seq_len_k, head_dim)
        value: (batch, num_heads, seq_len_k, head_dim)
        attn_mask: optional attention mask, broadcastable to (batch, num_heads, seq_len_q, seq_len_k)
        scale: custom scaling factor. If None, use 1/sqrt(head_dim)
        is_causal: whether to apply causal mask

    Returns:
        attn_output: (batch, num_heads, seq_len_q, head_dim)
    """

    B, H, Lq, D = query.shape
    _, _, Lk, _ = key.shape

    # print q, k, v shapes
    # print(f"Query shape: {query.shape}, Key shape: {key.shape}, Value shape: {value.shape}")

    if scale is None:
        scale = 1.0 / (D ** 0.5)

    # Step 1: Compute raw attention scores
    scores = np.matmul(query, key.transpose(0, 1, 3, 2)) * scale  # shape: (B, H, Lq, Lk)

    # Step 2: Apply masks
    if attn_mask is not None:
        scores = np.where(attn_mask == 0, float('-inf'), scores)

    if is_causal:
        causal_mask = np.tril(np.ones((Lq, Lk), dtype=bool))
        scores = np.where(~causal_mask, float('-inf'), scores)

    # Step 3: Softmax over last dim (Lk)
    attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)

    # Step 4: Compute output
    # value shape: (B, H, Lk, D)
    attn_output = np.matmul(attn_weights, value)  # shape: (B, H, Lq, D)

    return attn_output



if __name__ == "__main__":
    # Example usage
    batch_size = 4
    num_heads = 32
    head_dim = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_type = torch.float32
    is_causal = False
    
    warmup_runs = 2
    repeat_time = 10
    print(f"is_causal: {is_causal}")
    
    # for seq_len in {1024, 2048, 4096, 8192, 16384, 32768}:
    for seq_len in {1024, 2048, 4096, 8192}:
        flops = 4 * num_heads * batch_size * head_dim * seq_len * seq_len // (2 if is_causal else 1)

        query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=data_type, device=device)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=data_type, device=device)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=data_type, device=device)

        ref = functional_attention_torch(
            query, key, value, attn_mask=None, scale=None, is_causal=is_causal
        )
        
        attn_output = None
        # Warm-up runs
        for _ in range(warmup_runs):
            attn_output = functional_attention_torch(
                query, key, value, attn_mask=None, scale=None, is_causal=is_causal
            )
        
        time_starts = time.time()
        for _ in range(repeat_time):
            attn_output = functional_attention_torch(
                query, key, value, attn_mask=None, scale=None, is_causal=is_causal
            )
        elapsed = (time.time() - time_starts) / (repeat_time * 1.0)
        
        print("time taken:", time.time() - time_starts, "for", repeat_time, "runs")
        
        gflops = flops / elapsed / 1e9
        tflops = flops / elapsed / 1e12

        print(f"Time taken for seq_len {seq_len}: {elapsed} seconds")
        print(f'{seq_len} gflops:{gflops}, tflops:{tflops}')
        
        # Verify correctness
        max_diff = torch.max(torch.abs(attn_output - ref)).item()
        print(f"Max difference with reference: {max_diff}")


    print(attn_output.shape)  # Expected output shape: (2, 4, 5, 8)