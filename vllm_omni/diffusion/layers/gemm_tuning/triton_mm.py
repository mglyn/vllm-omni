# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Dense BF16 X[M,K] @ W[N,K].T with FP32 accumulation."""

import torch
from vllm.triton_utils import tl, triton

from .backends import Tile


@triton.jit
def _mm(
    x,
    w,
    y,
    rows,
    cols: tl.constexpr,
    depth: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    # Group neighboring rows tiles so consecutive programs can reuse W in L2.
    pid = tl.program_id(0)
    num_m, num_n = tl.cdiv(rows, block_m), tl.cdiv(cols, block_n)
    group = pid // (8 * num_n)
    first_m = group * 8
    group_m = tl.minimum(num_m - first_m, 8)
    offset = pid % (8 * num_n)
    m = (first_m + offset % group_m) * block_m + tl.arange(0, block_m)
    n = (offset // group_m) * block_n + tl.arange(0, block_n)
    k = tl.arange(0, block_k)
    a = x + m[:, None].to(tl.int64) * depth + k[None, :]
    b = w + n[None, :].to(tl.int64) * depth + k[:, None]
    acc = tl.full((block_m, block_n), 0, tl.float32)
    for block in range(tl.cdiv(depth, block_k)):
        valid_k = block * block_k + k < depth
        av = tl.load(a, mask=(m[:, None] < rows) & valid_k[None, :], other=0.0)
        bv = tl.load(b, mask=valid_k[:, None] & (n[None, :] < cols), other=0.0)
        acc = tl.dot(av, bv, acc)
        a += block_k
        b += block_k
    tl.store(
        y + m[:, None].to(tl.int64) * cols + n[None, :],
        acc.to(tl.bfloat16),
        mask=(m[:, None] < rows) & (n[None, :] < cols),
    )


def linear(x: torch.Tensor, weight: torch.Tensor, tile: Tile) -> torch.Tensor:
    n, k = weight.shape
    m = x.numel() // k
    out = x.new_empty((*x.shape[:-1], n))
    grid = (triton.cdiv(m, tile.block_m) * triton.cdiv(n, tile.block_n),)
    _mm[grid](
        x,
        weight,
        out,
        m,
        n,
        k,
        tile.block_m,
        tile.block_n,
        tile.block_k,
        num_stages=tile.num_stages,
        num_warps=tile.num_warps,
    )
    return out
