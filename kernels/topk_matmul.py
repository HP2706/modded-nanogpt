
import os
import torch

if not torch.cuda.is_available():
    os.environ['TRITON_INTERPRET'] = '1'

import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.core as core
from einops import rearrange



# Taken from https://github.com/fla-org/native-sparse-attention.git
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# Implements argsort based on bitonic sort.
# [What is bitonic sort?](https://en.wikipedia.org/wiki/Bitonic_sorter)

# Code adapted from https://github.com/triton-lang/triton/issues/3698#issuecomment-2067681396
import triton
import triton.language.core as core
from triton.language.standard import _log2, sum, zeros_like


@triton.jit
def _compare_and_swap(
    x,
    ids,
    flip,
    i: core.constexpr,
    n_dims: core.constexpr,
):
    n_outer: core.constexpr = x.numel >> n_dims
    shape: core.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]
    y = core.reshape(x, shape)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = core.arange(0, 2)[None, :, None]
    left = core.broadcast_to(sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = core.broadcast_to(sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = core.reshape(left, x.shape)
    right = core.reshape(right, x.shape)
    # idx
    y_idx = core.reshape(ids, shape)
    left_idx = core.broadcast_to(sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = core.broadcast_to(sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = core.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = core.reshape(right_idx, x.shape).to(y_idx.dtype)
    # actual compare-and-swap
    idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) != flip
    ret = ix ^ core.where(cond, ileft ^ iright, zeros_like(ix))
    new_ids = ids ^ core.where(cond, left_idx ^ right_idx, zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(
    x,
    ids,
    stage: core.constexpr,
    order: core.constexpr,
    n_dims: core.constexpr,
):
    '''
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    '''
    n_outer: core.constexpr = x.numel >> n_dims
    core.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: core.constexpr = [n_outer * 2**(n_dims - 1 - stage), 2, 2**stage]
        flip = core.reshape(core.broadcast_to(core.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in core.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def argsort(
    x,
    ids,
    dim: core.constexpr = None,
    descending: core.constexpr = core.CONSTEXPR_0,
):
    # handle default dimension or check that it is the most minor dim
    _dim: core.constexpr = len(x.shape) - 1 if dim is None else dim
    core.static_assert(_dim == len(x.shape) - 1, "only minor dimension is currently supported")
    # iteratively run bitonic merge-sort steps
    n_dims: core.constexpr = _log2(x.shape[_dim])

    for i in core.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending, n_dims)
    return x, ids


@triton.jit
def topk_matmul_kernel(
    a_ptr,  # Matrix A [M, K]
    b_ptr,  # Matrix B [K, N]
    stride_am, stride_ak, 
    stride_bk, stride_bn,
    out_indices_ptr,  # Output indices [M, TK]
    out_values_ptr,  # Output values [M, TK]
    stride_om, stride_otk,
    M: tl.constexpr,  # Common dimension
    N: tl.constexpr,  # Second matrix dimension
    K: tl.constexpr,  # Number of top values to keep
    TK: tl.constexpr,  # Number of top values to keep
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr, 
    GROUP_SIZE_M: tl.constexpr, 
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_tk = tl.arange(0, TK) 
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Fix the pointer arithmetic for output tensors

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    offs = tl.where(
        (tl.arange(0, BLOCK_SIZE_N)[:, None] >= 0) & (tl.arange(0, BLOCK_SIZE_M)[None, :] >= 0), 
        tl.arange(0, BLOCK_SIZE_N), 
        -1
    )
    
    sorted_values, sorted_indices = argsort(accumulator, offs, descending=True)
    
    # Create base mask for one row
    base_mask = tl.arange(0, TK) 
    row_indices = tl.arange(0, BLOCK_SIZE_M) 
    
    # create a mask for the top-k elements
    # this is a workaround to do base_mask.repeat(BLOCK_SIZE_M, 1)
    repeat_mask = tl.where(
        (row_indices[:, None] >= 0) & (base_mask[None, :] >= 0), 
        base_mask, 
        -1
    )

    gathered_values = tl.gather(sorted_values, repeat_mask, axis=1)
    gathered_indices = tl.gather(sorted_indices, repeat_mask, axis=1)
    
    # Store the gathered elements
    offs_am_for_out = offs_am[:, None]  # Shape: [BLOCK_SIZE_M, 1]
    offs_tk_for_out = offs_tk[None, :]  # Shape: [1, TK]
    out_indices_ptrs = out_indices_ptr + (offs_am_for_out * stride_om + offs_tk_for_out * stride_otk)
    out_values_ptrs = out_values_ptr + (offs_am_for_out * stride_om + offs_tk_for_out * stride_otk)
    
    tl.store(out_values_ptrs, gathered_values)
    tl.store(out_indices_ptrs, gathered_indices)

def topk_matmul(
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    k: int,
) -> torch.LongTensor:  # [M, k]
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Inner dimensions must match"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    #assert b.is_contiguous(), "Matrix B must be contiguous"
    
    # Make k a power of 2 for bitonic sort
    k_pow2 = triton.next_power_of_2(k)
    
    # Output tensor for indices
    indices = torch.zeros((M, k_pow2), device=a.device, dtype=torch.long)
    values = torch.zeros((M, k_pow2), device=a.device, dtype=a.dtype)
    
    # Define block sizes - these were missing in the original call
    BLOCK_SIZE_M = triton.next_power_of_2(M)
    BLOCK_SIZE_N = triton.next_power_of_2(N)
    BLOCK_SIZE_K = triton.next_power_of_2(K)
    GROUP_SIZE_M = triton.next_power_of_2(M)
    
    print("launching kernel")
    # Define grid function with the META parameter
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), 
    )
    # Launch kernel with all required parameters
    topk_matmul_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        out_indices_ptr=indices,
        out_values_ptr=values,
        stride_om=indices.stride(0),
        stride_otk=indices.stride(1),
        M=M,
        N=N,
        K=K,
        TK=k_pow2,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return values[:, :k], indices[:, :k]


def torch_topk_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    k: int,
) -> torch.LongTensor:
    return torch.topk(a @ b, k, dim=-1, sorted=True) # type: ignore


if __name__ == "__main__":
    M = 16
    N = 16
    K = 16
    TK = 4
    
    MAT_A = torch.randn((M, K), dtype=torch.float32)
    MAT_B = torch.randn((N, K), dtype=torch.float32)
    O = torch.zeros((M, TK), device=MAT_A.device, dtype=torch.int64)
    I = torch.arange(M*N).reshape(M, N).to(torch.float32)
    a, b = topk_matmul(MAT_A, MAT_B.T, TK)
    c, d = torch_topk_matmul(MAT_A, MAT_B.T, TK)
    
    # Sort both values and indices together
    sorted_indices = torch.argsort(a, dim=-1, descending=True)
    a = torch.gather(a, -1, sorted_indices)
    b = torch.gather(b, -1, sorted_indices)
    
    
    print(torch.norm(a-c).mean())
    print(b == d)

        
    