"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch, math
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner_qk_int8_pv_fp16(acc, l_i, m_i, q, q_scale, qo_len, kv_len,
                    K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn, 
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
            p = p.to(tl.float16)
            
            acc += tl.dot(p, v, out_dtype=tl.float16)   
            m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i

@triton.jit
def _attn_fwd_qk_int8_pv_fp16(Q, K, V, Q_scale, K_scale, Out, mask, Lse, 
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
              v_scale: tl.constexpr
              ):
    start_m = tl.program_id(0)

    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)  
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None] 
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
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
    acc, l_i, m_i = _attn_fwd_inner_qk_int8_pv_fp16(acc, l_i, m_i, q, q_scale, qo_len, kv_len, K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
                                    start_m, mask_ptrs, stride_maskn,
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  
                                    4 - STAGE, offs_m, offs_n 
                                    )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < qo_len))


@triton.jit
def compute_scale_vectorized(x, BLOCK_SIZE: tl.constexpr):
    """向量化的scale计算"""
    # 使用向量操作计算最大值
    abs_max = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for i in range(0, BLOCK_SIZE):
        val = tl.load(x + i).to(tl.float32)
        abs_max = tl.where(i == tl.arange(0, BLOCK_SIZE), tl.abs(val), abs_max)
    
    # 在向量中找最大值
    max_val = tl.max(abs_max, axis=0)
    
    # 计算scale
    safe_max = tl.maximum(max_val, 1e-8)
    scale = 127.0 / safe_max
    
    return scale

@triton.jit
def quantize_fp16_to_int8(x, scale):
    """量化FP16到INT8"""
    # 应用scale并四舍五入到最近的整数
    x_scaled = x * scale
    # 限制在INT8范围内 [-128, 127]
    x_quant = tl.clamp(x_scaled, 0.0, 127.0).to(tl.int8)
    # x_clipped = tl.minimum(tl.maximum(x_scaled, -128.0), 127.0)
    # 四舍五入并转换为INT8
    # x_quant = tl.math.round(x_clipped).to(tl.int8)
    return x_quant

@triton.jit
def compute_scale_fp16_to_int8(x, BLOCK_SIZE: tl.constexpr):
    """计算量化scale"""
    # 找到绝对值的最大值
    abs_max = tl.zeros([1], dtype=tl.float32)
    
    for i in range(0, BLOCK_SIZE):
        val = tl.load(x + i).to(tl.float32)
        abs_val = tl.abs(val)
        abs_max = tl.maximum(abs_max, abs_val)
    
    # 计算scale: 127.0 / abs_max
    # 防止除零
    safe_abs_max = tl.maximum(abs_max, 1e-8)
    scale = 127.0 / safe_abs_max
    
    return scale

@triton.jit
def _attn_fwd_inner_qk_int8_pv_int8(acc, l_i, m_i, q, q_scale, qo_len, kv_len,
                    K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn, 
                    start_m, mask_ptrs, stride_maskn,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr, v_scale: tl.constexpr
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

            # # block 内求 max(abs(x))
            abs_p = tl.abs(p)
            max_val = tl.max(abs_p, 0)                     # shape: [BLOCK_N]
            p_scale = max_val / 127.0                      # per-column scale [BLOCK_N]

            p = (p / p_scale)
            p = tl.clamp(p, -128, 127)
            p = tl.cast(p, tl.int8)
            
            # scale_v = p_scale * v_scale
            acc_f32 = tl.dot(p.to(tl.float32), v.to(tl.float32), out_dtype=tl.float32)

            acc += acc_f32

            m_i = m_ij

        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i

@triton.jit
def _attn_fwd_qk_int8_pv_int8(Q, K, V, Q_scale, K_scale, Out, mask, Lse, 
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
              v_scale: tl.constexpr
              ):
    start_m = tl.program_id(0)

    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)  
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None]
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
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
    acc, l_i, m_i = _attn_fwd_inner_qk_int8_pv_int8(acc, l_i, m_i, q, q_scale, qo_len, kv_len, K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
                                    start_m, mask_ptrs, stride_maskn,
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  
                                    4 - STAGE, offs_m, offs_n, v_scale
                                    )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < qo_len))

# 输入 是 q,k,v 都是int8， scale 是 float16
def atten_forward(forward_function, q, k, v, q_scale, k_scale, tensor_layout="HND", attn_mask=None, output_dtype=torch.float16, v_scale=0.0):
    BLOCK_M = 128
    BLOCK_N = 64
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

    lse = torch.empty([0], dtype=torch.float32, device='cpu')

    grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b)
    forward_function[grid](
        q, k, v, q_scale, k_scale, o, attn_mask, lse,
        stride_bz_q, stride_h_q, stride_seq_q, 
        stride_bz_k, stride_h_k, stride_seq_k,  
        stride_bz_v, stride_h_v, stride_seq_v,  
        stride_bz_o, stride_h_o, stride_seq_o,
        stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask,
        qo_len, kv_len,
        h_qo, num_kv_groups,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,  
        STAGE=stage,
        num_warps=4 if head_dim == 64 else 8,
        num_stages=3 if head_dim == 64 else 4, v_scale = v_scale)

    return o

# 输入还是 q,k,v 是fp32，用于测试精度
def atten_forward_triton(q, k, v, attn_mask=None):
    tensor_layout="HND"
    output_dtype=torch.float32

    # quantize q,k,v to int8
    q_scale = torch.max(torch.abs(q), dim=-1, keepdim=True)[0] / 127.0
    k_scale = torch.max(torch.abs(k), dim=-1, keepdim=True)[0] / 127.0
    v_scale = torch.max(torch.abs(v), dim=-1, keepdim=True)[0] / 127.0
    q_int8 = (q / q_scale).round().clamp(-128, 127).to(torch.int8)
    k_int8 = (k / k_scale).round().clamp(-128, 127).to(torch.int8)
    v_int8 = (v / v_scale).round().clamp(-128, 127).to(torch.int8)
    
    v_fp16 = v.to(torch.float16)

    o_int8_int8 = atten_forward(_attn_fwd_qk_int8_pv_int8, q_int8, k_int8, v_int8, q_scale.to(torch.float16), k_scale.to(torch.float16), tensor_layout=tensor_layout, attn_mask=attn_mask, output_dtype=output_dtype, v_scale=v_scale.to(torch.float16))
    o_int8_fp16 = atten_forward(_attn_fwd_qk_int8_pv_fp16, q_int8, k_int8, v_fp16, q_scale.to(torch.float16), k_scale.to(torch.float16), tensor_layout=tensor_layout, attn_mask=attn_mask, output_dtype=output_dtype, v_scale=v_scale.to(torch.float16))
    
    return o_int8_int8, o_int8_fp16


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
    B, H, Lq, Lk, D = 2, 4, 16, 16, 64
    query = torch.randn(B, H, Lq, D, device='cuda', dtype=torch.float32)
    key = torch.randn(B, H, Lk, D, device='cuda', dtype=torch.float32)
    value = torch.randn(B, H, Lk, D, device='cuda', dtype=torch.float32)

    # No mask
    attn_output_triton_int8_int8, attn_output_triton_int8_fp16 = atten_forward_triton(query, key, value)
    attn_output_torch = functional_attention_torch(query, key, value)

    # compute cosine similarity between outputs
    cos_sim_int8_int8 = torch.nn.functional.cosine_similarity(attn_output_triton_int8_int8, attn_output_torch)
    cos_sim_int8_fp16 = torch.nn.functional.cosine_similarity(attn_output_triton_int8_fp16, attn_output_torch)

    print("Cosine similarity (int8_int8):", cos_sim_int8_int8.mean().item())
    print("Cosine similarity (int8_fp16):", cos_sim_int8_fp16.mean().item())
    
    print("Output difference (int8_int8):", torch.max(torch.abs(attn_output_triton_int8_int8 - attn_output_torch)).item())
    print("Output difference (int8_fp16):", torch.max(torch.abs(attn_output_triton_int8_fp16 - attn_output_torch)).item())