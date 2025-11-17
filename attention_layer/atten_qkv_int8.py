import torch, math
import triton
import triton.language as tl

@triton.jit
def quant_per_block_int8_kernel(Input, Output, Scale, L,
                                stride_iz, stride_ih, stride_in,
                                stride_oz, stride_oh, stride_on,
                                stride_sz, stride_sh,
                                sm_scale,
                                C: tl.constexpr, BLK: tl.constexpr):
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    x *= sm_scale
    scale = tl.max(tl.abs(x)) / 127.
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)


def per_block_int8(q, k, v, km=None, BLKQ=128, BLKK=64, BLKV=64, sm_scale=None, tensor_layout="HND"):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    v_int8 = torch.empty(v.shape, dtype=torch.int8, device=v.device)

    if km is not None:
        k = k - km  # 为什么这里可以直接 使用k- km替代后面的k？因为 softmax (x - m) = softmax(x), 所以直接在量化前减去km是可以的。

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(1), q_int8.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(1), k_int8.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_vo, stride_h_vo, stride_seq_vo = v_int8.stride(0), v_int8.stride(1), v_int8.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape # kv 都是一样的

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(2), q_int8.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(2), k_int8.stride(1)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_vo, stride_h_vo, stride_seq_vo = v_int8.stride(0), v_int8.stride(2), v_int8.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    q_scale = torch.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ), device=q.device, dtype=torch.float32) # q = b,h,len,d, 正常是每个，量化之后的scale应该是 b,h,len/block。
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK), device=q.device, dtype=torch.float32)
    v_scale = torch.empty((b, h_kv, (kv_len + BLKV - 1) // BLKV), device=q.device, dtype=torch.float32)

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    grid = ((qo_len + BLKQ - 1) // BLKQ, h_qo, b)
    quant_per_block_int8_kernel[grid](
        q, q_int8, q_scale, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_qo, stride_h_qo, stride_seq_qo,
        q_scale.stride(0), q_scale.stride(1),
        sm_scale=(sm_scale * 1.44269504),
        C=head_dim, BLK=BLKQ
    )

    grid = ((kv_len + BLKK - 1) // BLKK, h_kv, b)
    quant_per_block_int8_kernel[grid](
        k, k_int8, k_scale, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_ko, stride_h_ko, stride_seq_ko,
        k_scale.stride(0), k_scale.stride(1),
        sm_scale=1.0,
        C=head_dim, BLK=BLKK
    )
 
    grid = ((kv_len + BLKV - 1) // BLKV, h_kv, b)
    quant_per_block_int8_kernel[grid](
        v, v_int8, v_scale, kv_len,
        stride_bz_v, stride_h_v, stride_seq_v,
        stride_bz_vo, stride_h_vo, stride_seq_vo,
        v_scale.stride(0), v_scale.stride(1),
        sm_scale=1.0,
        C=head_dim, BLK=BLKV
    )

    return q_int8, q_scale, k_int8, k_scale, v_int8, v_scale

@triton.jit
def _attn_fwd_inner_qkv_int8(acc, l_i, m_i, q, q_scale, qo_len, kv_len,
                    K_ptrs, K_scale_ptr, V_ptrs, V_scale_ptr, V_mean_ptr, stride_kn, stride_vn, 
                    start_m, mask_ptrs, stride_maskn,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  
                    ):
    lo, hi = 0, kv_len
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_block = None
        skip = False
        if mask_ptrs is not None:
            if mask_ptrs.dtype.element_ty == tl.int1:
                mask_block = tl.load(mask_ptrs + start_n * stride_maskn, mask=(offs_m[:, None] < qo_len) & (offs_n[None, :] < kv_len - start_n), other=False)
                if tl.max(mask_block) == 0:
                    skip = True
            else:
                mask_block = tl.load(mask_ptrs + start_n * stride_maskn, mask=(offs_m[:, None] < qo_len) & (offs_n[None, :] < kv_len - start_n), other=-1.0e6)
        if not skip:
            k_mask = offs_n[None, :] < (kv_len - start_n)
            k = tl.load(K_ptrs, mask=k_mask)
            k_scale = tl.load(K_scale_ptr)

            qk = tl.dot(q, k).to(tl.float32) * (q_scale * k_scale)
            
            if mask_block is not None:
                if mask_block.dtype == tl.int1:
                    qk = qk + tl.where(mask_block, 0, -1.0e6)
                else:
                    qk = qk + mask_block
            else:
                qk += tl.where(k_mask, 0, -1.0e6)

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            
            acc = acc * alpha[:, None]
            
            v = tl.load(V_ptrs, mask = offs_n[:, None] < (kv_len - start_n))
            v_scale = tl.load(V_scale_ptr)
            p = p.to(tl.float32)
            
            # 将p转换为int8
            # absmax = tl.maximum(tl.max(tl.abs(p)), 1e-10)
            # p_scale = absmax / 127

            # p_q = p * (127 / absmax)
            # p_q = tl.extra.cuda.libdevice.round(p_q).to(tl.int8)

            v_mean = tl.load(V_mean_ptr, mask=offs_n[:, None] < (kv_len - start_n))
            v_fp = (v.to(tl.float32)) * v_scale + v_mean
            
            acc += tl.dot(p, v_fp, out_dtype=tl.float32)
            m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vn
        V_scale_ptr += 1
        V_mean_ptr += BLOCK_N
    return acc, l_i, m_i

@triton.jit
def _attn_fwd_qkv_int8(Q, K, V, Q_scale, K_scale, V_scale, V_mean, Out, mask, Lse, 
              stride_qz, stride_qh, stride_qn,
              stride_kz, stride_kh, stride_kn,  
              stride_vz, stride_vh, stride_vn,  
              stride_oz, stride_oh, stride_on, 
              stride_maskz, stride_maskh, stride_maskm, stride_maskn,
              qo_len, kv_len, H: tl.constexpr, num_kv_groups: tl.constexpr,
              HEAD_DIM: tl.constexpr,  
              BLOCK_M: tl.constexpr,  
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr,
              RETURN_LSE: tl.constexpr,
              ):
    start_m = tl.program_id(0)

    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)  
    v_mean_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * kv_len
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None] 
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    V_scale_ptr = V_scale + k_scale_offset
    V_mean_ptr = V_mean + v_mean_offset + offs_n[:, None]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]
    if mask is None:
        mask_ptrs = None
    else:
        mask_ptrs = mask + (off_z * stride_maskz + off_h * stride_maskh) + offs_m[:, None] * stride_maskm + offs_n[None, :] * stride_maskn

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    q = tl.load(Q_ptrs, mask = offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)

    acc, l_i, m_i = _attn_fwd_inner_qkv_int8(acc, l_i, m_i, q, q_scale, qo_len, kv_len, K_ptrs, K_scale_ptr, V_ptrs, V_scale_ptr, V_mean_ptr, stride_kn, stride_vn,
                                    start_m, mask_ptrs, stride_maskn,
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  
                                    4 - STAGE, offs_m, offs_n 
                                    )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < qo_len))

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask = (offs_m < qo_len))


def forward_qkv_int8(q, k, v, q_scale, k_scale, v_scale, v_mean, tensor_layout="HND", attn_mask=None, output_dtype=torch.float16, return_lse=False):
    BLOCK_M = 64
    BLOCK_N = 32
    stage = 1

    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")

    if attn_mask is not None:
        stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask = attn_mask.stride(0), attn_mask.stride(1), attn_mask.stride(2), attn_mask.stride(3)
    else:
        stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask = 0, 0, 0, 0

    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv

    if return_lse:
        lse = torch.empty([b, h_qo, qo_len], dtype=torch.float32, device=q.device)
    else:
        lse = torch.empty([0], dtype=torch.float32, device='cpu')

    grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b)
    _attn_fwd_qkv_int8[grid](
        q, k, v, q_scale, k_scale, v_scale, v_mean, o, attn_mask, lse,
        stride_bz_q, stride_h_q, stride_seq_q, 
        stride_bz_k, stride_h_k, stride_seq_k,  
        stride_bz_v, stride_h_v, stride_seq_v,  
        stride_bz_o, stride_h_o, stride_seq_o,
        stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask,
        qo_len, kv_len,
        h_qo, num_kv_groups,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,  
        STAGE=stage, RETURN_LSE=return_lse,
        num_warps=4 if head_dim == 64 else 8,
        num_stages=3 if head_dim == 64 else 4)

    return o, lse


def atten_forward_triton_int8(q, k, v, attn_mask=None):
    tensor_layout="HND"
    output_dtype=torch.float32
    km = None

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")
    # if sm_scale is None:
    sm_scale = 1.0 / (head_dim_og ** 0.5) # 这里会将默认的缩放因子放到量化的scale中
    
    import time
    time_start = time.time()
    
    v_mean = v.float().mean(dim=-1, keepdim=True)
    v = v - v_mean
    
    # print v and v_mean shape
    q_int8, q_scale, k_int8, k_scale, v_int8, v_scale = per_block_int8(q, k, v, km=km, sm_scale=sm_scale, tensor_layout=tensor_layout)
    # print(f"v shape: {v.shape}, v_mean shape: {v_mean.shape}, v_scale shape: {v_scale.shape}, v_int8 shape: {v_int8.shape}")
    
    # v_scale2 = v_scale.unsqueeze(-1).repeat(1, 1, 1, 64).reshape(2, 16, 4096)
    
    # print(f"v_scale2 shape: {v_scale2.shape}")
    # v_recovery = v_int8.to(torch.float32) * v_scale2.unsqueeze(-1) + v_mean
    # # compute cosine similarity between v and v_recovery
    # cos_sim_out_int8 = torch.nn.functional.cosine_similarity(v_recovery, v)
    # print("Cosine similarity (v_recovery):", cos_sim_out_int8.mean().item())
    
    # torch.cuda.synchronize()
    # time_end = time.time()
    # print(f"per block int8 quant time: {(time_end - time_start) * 1000 :.6f} ms")
    time_start = time.time()
    
    o, lse = forward_qkv_int8(q_int8, k_int8, v_int8, q_scale, k_scale, v_scale, v_mean, tensor_layout=tensor_layout, output_dtype=output_dtype, attn_mask=attn_mask, return_lse=False)
      # torch.cuda.synchronize()
    # time_end = time.time()
    # print(f"per block int8 compute time: {(time_end - time_start) * 1000 :.6f} ms")
    return o

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


if __name__ == "__main__":
    # random seed
    torch.manual_seed(42)

    # Test the attention implementation
    B, H, Lq, Lk, D = 2, 16, 4096, 4096, 128
    query = torch.randn(B, H, Lq, D, device='cuda', dtype=torch.float32)
    key = torch.randn(B, H, Lk, D, device='cuda', dtype=torch.float32)
    value = torch.randn(B, H, Lk, D, device='cuda', dtype=torch.float32)

    # No mask
    # attn_output_triton_int8_int8, attn_output_triton_int8_fp16, attn_output_triton_fp16_fp16 = atten_forward_triton(query, key, value)
    attn_output_torch = functional_attention_torch(query, key, value)
    out_int8 = atten_forward_triton_int8(query, key, value)

    # # compute cosine similarity between outputs
    # cos_sim_int8_int8 = torch.nn.functional.cosine_similarity(attn_output_triton_int8_int8, attn_output_torch)
    # cos_sim_int8_fp16 = torch.nn.functional.cosine_similarity(attn_output_triton_int8_fp16, attn_output_torch)
    # cos_sim_fp16_fp16 = torch.nn.functional.cosine_similarity(attn_output_triton_fp16_fp16, attn_output_torch)
    cos_sim_out_int8 = torch.nn.functional.cosine_similarity(out_int8, attn_output_torch)
    
    print("Cosine similarity (out_int8):", cos_sim_out_int8.mean().item())