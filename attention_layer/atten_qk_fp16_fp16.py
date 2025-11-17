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

def per_block_int8(q, k, km=None, BLKQ=128, BLKK=64, sm_scale=None, tensor_layout="HND"):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if km is not None:
        k = k - km  # 为什么这里可以直接 使用k- km替代后面的k？因为 softmax (x - m) = softmax(x), 所以直接在量化前减去km是可以的。

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(1), q_int8.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(1), k_int8.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(2), q_int8.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(2), k_int8.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    q_scale = torch.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ), device=q.device, dtype=torch.float32) # q = b,h,len,d, 正常是每个，量化之后的scale应该是 b,h,len, 但是Len不一定对齐了，所以需要对齐成 BLKQ大小，这个大小是和后面 attention计算时，一个block内部计算q的长度相对应的。
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK), device=q.device, dtype=torch.float32)

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

    return q_int8, q_scale, k_int8, k_scale



@triton.jit
def _attn_fwd_inner_qk_fp16_pv_fp16(acc, l_i, m_i, q, qo_len, kv_len,
                    K_ptrs, V_ptrs, stride_kn, stride_vn, 
                    start_m, mask_ptrs, stride_maskn,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  
                    ):
    lo, hi = 0, kv_len
    for start_n in range(lo, hi, BLOCK_N): # 计算时，B,H是一一对应的，seq一次计算N块
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
            # k_scale = tl.load(K_scale_ptr)

            qk = tl.dot(q, k, out_dtype=tl.float32) * 0.0883883 #* (q_scale * k_scale)
            
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
            # p = p.to(tl.float16)
            
            acc += tl.dot(p, v, out_dtype=tl.float32)
            m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        # K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i

@triton.jit
def _attn_fwd_qk_fp16_pv_fp16(Q, K, V, Out, mask, Lse, 
              stride_qz, stride_qh, stride_qn, # q的 B,H,len
              stride_kz, stride_kh, stride_kn, # k的 B,H,len
              stride_vz, stride_vh, stride_vn, # v的 B,H,len
              stride_oz, stride_oh, stride_on, # out的 B,H,len
              stride_maskz, stride_maskh, stride_maskm, stride_maskn,
              qo_len, kv_len, H: tl.constexpr, num_kv_groups: tl.constexpr,
              HEAD_DIM: tl.constexpr,  
              BLOCK_M: tl.constexpr,  
              BLOCK_N: tl.constexpr,  
              STAGE: tl.constexpr,
              RETURN_LSE: tl.constexpr,
              ):
    start_m = tl.program_id(0)            # inside q_len // BLOCK_M

    off_z = tl.program_id(2).to(tl.int64) # batch index
    off_h = tl.program_id(1).to(tl.int64) # head index

    # q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
    # k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)  
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M) # 一次处理M块
    offs_n = tl.arange(0, BLOCK_N)                     # 一次处理N块
    offs_k = tl.arange(0, HEAD_DIM)                    # head_dim # 这个对应于 D
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :] # 取出 MxD的块
    # Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None] # 对于group attention，来说，要找到正确的head组 提取出NxD的块
    # K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :] # 输出的 MxD块
    if mask is None:
        mask_ptrs = None
    else:
        mask_ptrs = mask + (off_z * stride_maskz + off_h * stride_maskh) + offs_m[:, None] * stride_maskm + offs_n[None, :] * stride_maskn

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32) # 对应输出的 M x D 块
    
    q = tl.load(Q_ptrs, mask = offs_m[:, None] < qo_len)
    # q_scale = tl.load(Q_scale_ptr)
    
    # acc, l_i, m_i, q, q_scale, qo_len, kv_len,
    # K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn, 
    # start_m, mask_ptrs, stride_maskn,
    # BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  
    # STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  
    acc, l_i, m_i = _attn_fwd_inner_qk_fp16_pv_fp16(acc, l_i, m_i, q, qo_len, kv_len, K_ptrs, V_ptrs, stride_kn, stride_vn,
                                    start_m, mask_ptrs, stride_maskn,
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  
                                    4 - STAGE, offs_m, offs_n 
                                    )
    acc = acc / l_i[:, None] # 相当于 softmax 滞后除法。
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < qo_len))

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask = (offs_m < qo_len))

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
              RETURN_LSE: tl.constexpr,
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

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask = (offs_m < qo_len))

def forward_qk_int8_pv_fp16(q, k, v, q_scale, k_scale, tensor_layout="HND", attn_mask=None, output_dtype=torch.float16, return_lse=False):
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

    if return_lse:
        lse = torch.empty([b, h_qo, qo_len], dtype=torch.float32, device=q.device)
    else:
        lse = torch.empty([0], dtype=torch.float32, device='cpu')

    grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b)
    _attn_fwd_qk_int8_pv_fp16[grid](
        q, k, v, q_scale, k_scale, o, attn_mask, lse,
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


def forward_qk_fp16_pv_fp16(q, k, v, tensor_layout="HND", attn_mask=None, output_dtype=torch.float32, return_lse=False):
    BLOCK_M = 16
    BLOCK_N = 16
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

    grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b) # grid 是按照 q来划分的， N是内部的for循环，对应的是 b, h, len_k / BLOCK_N的划分，其中，按照规则，b和h是对应的，类似于矩阵乘法。
    _attn_fwd_qk_fp16_pv_fp16[grid](
        q, k, v, o, attn_mask, lse,
        stride_bz_q, stride_h_q, stride_seq_q, 
        stride_bz_k, stride_h_k, stride_seq_k,  
        stride_bz_v, stride_h_v, stride_seq_v,  
        stride_bz_o, stride_h_o, stride_seq_o,
        stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask,
        qo_len, kv_len,
        h_qo, num_kv_groups,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,   # q_split, N, head_dim # 其中N 是
        STAGE=stage, RETURN_LSE=return_lse,
        num_warps=4 if head_dim == 64 else 8,
        num_stages=3 if head_dim == 64 else 4)

    return o, lse


@triton.jit
def _attn_fwd_inner_qk_fp16_pv_fp16_v2(acc, l_i, m_i, q, qo_len, kv_len,
                    K_ptrs, V_ptrs, stride_kn, stride_vn, 
                    start_m, mask_ptrs, stride_maskn,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  
                    ):
    lo, hi = 0, kv_len
    for start_n in range(lo, hi, BLOCK_N): # 计算时，B,H是一一对应的，seq一次计算N块
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
            # k_scale = tl.load(K_scale_ptr)

            qk = tl.dot(q, k, out_dtype=tl.float16) * 0.0883883 #* (q_scale * k_scale)
            
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
            
            alpha = alpha.to(tl.float16)
            acc = acc * alpha[:, None]
            
            v = tl.load(V_ptrs, mask = offs_n[:, None] < (kv_len - start_n))
            p = p.to(tl.float16)
            
            acc += tl.dot(p, v, out_dtype=tl.float16)
            m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        # K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i

@triton.jit
def _attn_fwd_qk_fp16_pv_fp16_v2(Q, K, V, Out, mask, Lse, 
              stride_qz, stride_qh, stride_qn, # q的 B,H,len
              stride_kz, stride_kh, stride_kn, # k的 B,H,len
              stride_vz, stride_vh, stride_vn, # v的 B,H,len
              stride_oz, stride_oh, stride_on, # out的 B,H,len
              stride_maskz, stride_maskh, stride_maskm, stride_maskn,
              qo_len, kv_len, H: tl.constexpr, num_kv_groups: tl.constexpr,
              HEAD_DIM: tl.constexpr,  
              BLOCK_M: tl.constexpr,  
              BLOCK_N: tl.constexpr,  
              STAGE: tl.constexpr,
              RETURN_LSE: tl.constexpr,
              ):
    start_m = tl.program_id(0)            # inside q_len // BLOCK_M

    off_z = tl.program_id(2).to(tl.int64) # batch index
    off_h = tl.program_id(1).to(tl.int64) # head index

    # q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
    # k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)  
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M) # 一次处理M块
    offs_n = tl.arange(0, BLOCK_N)                     # 一次处理N块
    offs_k = tl.arange(0, HEAD_DIM)                    # head_dim # 这个对应于 D
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :] # 取出 MxD的块
    # Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None] # 对于group attention，来说，要找到正确的head组 提取出NxD的块
    # K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :] # 输出的 MxD块
    if mask is None:
        mask_ptrs = None
    else:
        mask_ptrs = mask + (off_z * stride_maskz + off_h * stride_maskh) + offs_m[:, None] * stride_maskm + offs_n[None, :] * stride_maskn

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float16) # 对应输出的 M x D 块
    
    q = tl.load(Q_ptrs, mask = offs_m[:, None] < qo_len)
    # q_scale = tl.load(Q_scale_ptr)
    
    # acc, l_i, m_i, q, q_scale, qo_len, kv_len,
    # K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn, 
    # start_m, mask_ptrs, stride_maskn,
    # BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  
    # STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  
    acc, l_i, m_i = _attn_fwd_inner_qk_fp16_pv_fp16_v2(acc, l_i, m_i, q, qo_len, kv_len, K_ptrs, V_ptrs, stride_kn, stride_vn,
                                    start_m, mask_ptrs, stride_maskn,
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  
                                    4 - STAGE, offs_m, offs_n 
                                    )
    acc = acc / l_i[:, None] # 相当于 softmax 滞后除法。
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < qo_len))

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask = (offs_m < qo_len))


def forward_qk_fp16_pv_fp16_v2(q, k, v, tensor_layout="HND", attn_mask=None, output_dtype=torch.float32, return_lse=False):
    BLOCK_M = 128
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

    grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b) # grid 是按照 q来划分的， N是内部的for循环，对应的是 b, h, len_k / BLOCK_N的划分，其中，按照规则，b和h是对应的，类似于矩阵乘法。
    _attn_fwd_qk_fp16_pv_fp16_v2[grid](
        q, k, v, o, attn_mask, lse,
        stride_bz_q, stride_h_q, stride_seq_q, 
        stride_bz_k, stride_h_k, stride_seq_k,  
        stride_bz_v, stride_h_v, stride_seq_v,  
        stride_bz_o, stride_h_o, stride_seq_o,
        stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask,
        qo_len, kv_len,
        h_qo, num_kv_groups,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,   # q_split, N, head_dim # 其中N 是
        STAGE=stage, RETURN_LSE=return_lse,
        num_warps=4 if head_dim == 64 else 8,
        num_stages=3 if head_dim == 64 else 4)

    return o, lse



def atten_forward_triton_new_sageatten(q, k, v, attn_mask=None):
    v_fp16 = v.to(torch.float16)
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
    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, km=km, sm_scale=sm_scale, tensor_layout=tensor_layout)
    # torch.cuda.synchronize()
    # time_end = time.time()
    # print(f"per block int8 quant time: {(time_end - time_start) * 1000 :.6f} ms")
    time_start = time.time()
    o, lse = forward_qk_int8_pv_fp16(q_int8, k_int8, v_fp16, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=output_dtype, attn_mask=attn_mask, return_lse=False)
      # torch.cuda.synchronize()
    # time_end = time.time()
    # print(f"per block int8 compute time: {(time_end - time_start) * 1000 :.6f} ms")
    return o


def atten_forward_triton_new_sageatten_fp16(q, k, v, attn_mask=None):
    
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

    o_fp16, _ = forward_qk_fp16_pv_fp16(q, k, v, tensor_layout=tensor_layout, output_dtype=output_dtype, attn_mask=attn_mask, return_lse=False)
  
    return o_fp16

def atten_forward_triton_new_sageatten_fp16_v2(q, k, v, attn_mask=None):
    q = q.to(torch.float16)
    k = k.to(torch.float16)
    v = v.to(torch.float16)
    tensor_layout="HND"
    output_dtype=torch.float16
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

    o_fp16, _ = forward_qk_fp16_pv_fp16_v2(q, k, v, tensor_layout=tensor_layout, output_dtype=output_dtype, attn_mask=attn_mask, return_lse=False)
  
    return o_fp16

# 输入还是 q,k,v 是fp32，用于测试精度
# def atten_forward_triton(q, k, v, attn_mask=None):
#     tensor_layout="HND"
#     output_dtype=torch.float32

#     # quantize q,k,v to int8
#     q_scale = torch.max(torch.abs(q), dim=-1, keepdim=True)[0] / 127.0
#     k_scale = torch.max(torch.abs(k), dim=-1, keepdim=True)[0] / 127.0
#     v_scale = torch.max(torch.abs(v), dim=-1, keepdim=True)[0] / 127.0
#     q_int8 = (q / q_scale).round().clamp(-128, 127).to(torch.int8)
#     k_int8 = (k / k_scale).round().clamp(-128, 127).to(torch.int8)
#     v_int8 = (v / v_scale).round().clamp(-128, 127).to(torch.int8)
 
#     q_fp16 = q.to(torch.float16)
#     k_fp16 = k.to(torch.float16)
#     v_fp16 = v.to(torch.float16)

#     o_int8_int8 = atten_forward(_attn_fwd_qk_int8_pv_int8, q_int8, k_int8, v_int8, q_scale.to(torch.float16), k_scale.to(torch.float16), v_scale.to(torch.float16), tensor_layout=tensor_layout, attn_mask=attn_mask, output_dtype=output_dtype)
#     o_int8_fp16 = atten_forward(_attn_fwd_qk_int8_pv_fp16, q_int8, k_int8, v_fp16, q_scale.to(torch.float16), k_scale.to(torch.float16), v_scale.to(torch.float16), tensor_layout=tensor_layout, attn_mask=attn_mask, output_dtype=output_dtype)
#     o_fp16_fp16 = atten_forward(_attn_fwd_qk_fp16_pv_fp16, q_fp16, k_fp16, v_fp16, q_scale.to(torch.float16), k_scale.to(torch.float16), v_scale.to(torch.float16), tensor_layout=tensor_layout, attn_mask=attn_mask, output_dtype=output_dtype)
#     return o_int8_int8, o_int8_fp16, o_fp16_fp16


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
    
    v_fp16 = value.to(torch.float16)
    tensor_layout="HND"
    output_dtype=torch.float32
    km = None

    head_dim_og = query.size(-1)

    if head_dim_og < 64:
        query = torch.nn.functional.pad(query, (0, 64 - head_dim_og))
        key = torch.nn.functional.pad(key, (0, 64 - head_dim_og))
        value = torch.nn.functional.pad(value, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        query = torch.nn.functional.pad(query, (0, 128 - head_dim_og))
        key = torch.nn.functional.pad(key, (0, 128 - head_dim_og))
        value = torch.nn.functional.pad(value, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")
    # if sm_scale is None:
    sm_scale = 1.0 / (head_dim_og ** 0.5) # 这里会将默认的缩放因子放到量化的scale中
    
    import time
    time_start = time.time()
    q_int8, q_scale, k_int8, k_scale = per_block_int8(query, key, km=km, sm_scale=sm_scale, tensor_layout=tensor_layout)
    # torch.cuda.synchronize()
    # time_end = time.time()
    # print(f"per block int8 quant time: {(time_end - time_start) * 1000 :.6f} ms")
    time_start = time.time()
    out_int8, lse = forward_qk_int8_pv_fp16(q_int8, k_int8, v_fp16, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=output_dtype)
      # torch.cuda.synchronize()
    
    # out_int8 = atten_forward_triton_new_sageatten(query, key, value)
    out_fp16 = atten_forward_triton_new_sageatten_fp16(query, key, value)
    out_fp16_v2 = atten_forward_triton_new_sageatten_fp16_v2(query, key, value).to(torch.float32)

    # # compute cosine similarity between outputs
    # cos_sim_int8_int8 = torch.nn.functional.cosine_similarity(attn_output_triton_int8_int8, attn_output_torch)
    # cos_sim_int8_fp16 = torch.nn.functional.cosine_similarity(attn_output_triton_int8_fp16, attn_output_torch)
    # cos_sim_fp16_fp16 = torch.nn.functional.cosine_similarity(attn_output_triton_fp16_fp16, attn_output_torch)
    cos_sim_out_int8 = torch.nn.functional.cosine_similarity(out_int8, attn_output_torch)
    cos_sim_out_fp16 = torch.nn.functional.cosine_similarity(out_fp16, attn_output_torch)
    cos_sim_out_fp16_v2 = torch.nn.functional.cosine_similarity(out_fp16_v2, attn_output_torch)
    
    print("Cosine similarity (out_int8):", cos_sim_out_int8.mean().item())
    print("Cosine similarity (out_fp16):", cos_sim_out_fp16.mean().item())
    print("Cosine similarity (out_fp16_v2):", cos_sim_out_fp16_v2.mean().item())
    
    

    # # print first 10 results
    # print("Results attn_output_triton_int8_int8 :", attn_output_triton_int8_int8.flatten()[0:25])
    # print("Results attn_output_triton_int8_fp16 :", attn_output_triton_int8_fp16.flatten()[0:25])
    # print("Results attn_output_triton_fp16_fp16 :", attn_output_triton_fp16_fp16.flatten()[0:25])
    # print("Results attn_output_triton_new_sageatten :", attn_output_triton_new_sageatten.flatten()[0:25])
    # print("Results attn_output_torch :", attn_output_torch.flatten()[0:25])

    # print("Cosine similarity (int8_int8):", cos_sim_int8_int8.mean().item())
    # print("Cosine similarity (int8_fp16):", cos_sim_int8_fp16.mean().item())
    # print("Cosine similarity (fp16_fp16):", cos_sim_fp16_fp16.mean().item())
    # print("Cosine similarity (new_sageatten):", cos_sim_new_sageatten.mean().item())

    #  # print max absolute difference

    # print("Output difference (int8_int8):", torch.max(torch.abs(attn_output_triton_int8_int8 - attn_output_torch)).item())
    # print("Output difference (int8_fp16):", torch.max(torch.abs(attn_output_triton_int8_fp16 - attn_output_torch)).item())
    # print("Output difference (fp16_fp16):", torch.max(torch.abs(attn_output_triton_fp16_fp16 - attn_output_torch)).item())
    # print("Output difference (new_sageatten):", torch.max(torch.abs(attn_output_triton_new_sageatten - attn_output_torch)).item())


    # performance test
    import time
    
    loop_run = 100
    _ = atten_forward_triton_new_sageatten(query, key, value)
    sage_atten_time = 0.0
    for _ in range(loop_run):
        start = time.time()
        _ = atten_forward_triton_new_sageatten(query, key, value)
        torch.cuda.synchronize()
        end = time.time()
        
        sage_atten_time += (end - start)
        
    # _ = atten_forward_triton_new_sageatten(query, key, value)
    out_int8, lse = forward_qk_int8_pv_fp16(q_int8, k_int8, v_fp16, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=output_dtype)
    sage_atten_time_compute = 0.0
    for _ in range(loop_run):
        start = time.time()
        # _ = atten_forward_triton_new_sageatten(query, key, value)
        out_int8, lse = forward_qk_int8_pv_fp16(q_int8, k_int8, v_fp16, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=output_dtype)
        torch.cuda.synchronize()
        end = time.time()
        
        sage_atten_time_compute += (end - start)
        
    _ = atten_forward_triton_new_sageatten_fp16(query, key, value)
    sage_atten_time_fp16 = 0.0
    for _ in range(loop_run):
        start = time.time()
        _ = atten_forward_triton_new_sageatten_fp16(query, key, value)
        torch.cuda.synchronize()
        end = time.time()
        
        sage_atten_time_fp16 += (end - start)
    
    _ = atten_forward_triton_new_sageatten_fp16_v2(query, key, value)
    sage_atten_time_fp16_v2 = 0.0
    for _ in range(loop_run):
        start = time.time()
        _ = atten_forward_triton_new_sageatten_fp16_v2(query, key, value)
        torch.cuda.synchronize()
        end = time.time()
        
        sage_atten_time_fp16_v2 += (end - start)
    
    _ = functional_attention_torch(query, key, value)
    naive_atten_time = 0.0
    for _ in range(loop_run):
        start = time.time()
        _ = functional_attention_torch(query, key, value)
        torch.cuda.synchronize()
        end = time.time()
        
        naive_atten_time += (end - start)
    
    print(f"SageAtten time: {sage_atten_time / loop_run * 1000:.2f} ms")
    print(f"SageAtten compute time: {sage_atten_time_compute / loop_run * 1000:.2f} ms")
    print(f"Naive Atten time: {naive_atten_time / loop_run * 1000:.2f} ms")
    print(f"SageAtten FP16 time: {sage_atten_time_fp16 / loop_run * 1000:.2f} ms")
    print(f"SageAtten FP16 v2 time: {sage_atten_time_fp16_v2 / loop_run * 1000:.2f} ms")
        