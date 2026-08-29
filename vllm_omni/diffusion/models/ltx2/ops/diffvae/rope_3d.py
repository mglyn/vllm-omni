# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bit-exact Triton Q/K scaling and paired 3D RoPE for LTX-2.5 DiffVAE."""

from __future__ import annotations

import logging
import math

import torch
from vllm.triton_utils import tl, triton

from ..numerics import add_rn_f32, mul_rn_f32, round_bf16_to_fp32
from ..platform import is_ltx2_ops_eligible

_BLOCK_PAIRS = 256
_TABLE_CACHE_MAX = 16
_TABLE_CACHE: dict[
    tuple[torch.device, int, int, int, tuple[int, int, int], float],
    tuple[tuple[torch.Tensor, torch.Tensor], ...],
] = {}
_FAILED_DEVICES: set[int | None] = set()
_VERIFIED_CONTRACTS: set[tuple[int | None, tuple[int, int, int], float, float]] = set()

logger = logging.getLogger(__name__)


@triton.jit
def _qk_scale_rope_3d_kernel(
    query_output_ptr,
    key_output_ptr,
    query_ptr,
    key_ptr,
    cos_t_ptr,
    sin_t_ptr,
    cos_h_ptr,
    sin_h_ptr,
    cos_w_ptr,
    sin_w_ptr,
    num_pairs,
    tokens_per_batch,
    height,
    width,
    heads,
    query_scale,
    pairs_per_head: tl.constexpr,
    pairs_t: tl.constexpr,
    pairs_h: tl.constexpr,
    block_pairs: tl.constexpr,
):
    pair_offsets = tl.program_id(0).to(tl.int64) * block_pairs + tl.arange(0, block_pairs)
    valid = pair_offsets < num_pairs
    pair_in_head = pair_offsets % pairs_per_head
    rows = pair_offsets // pairs_per_head
    tokens = (rows // heads) % tokens_per_batch
    pos_w = tokens % width
    pos_h = (tokens // width) % height
    pos_t = tokens // (height * width)

    is_t = pair_in_head < pairs_t
    is_h = (pair_in_head >= pairs_t) & (pair_in_head < pairs_t + pairs_h)
    is_w = ~(is_t | is_h)
    pair_t = pair_in_head
    pair_h = pair_in_head - pairs_t
    pair_w = pair_in_head - pairs_t - pairs_h
    pairs_w: tl.constexpr = pairs_per_head - pairs_t - pairs_h
    cos_t = tl.load(cos_t_ptr + pos_t * pairs_t + pair_t, mask=valid & is_t, other=0.0)
    sin_t = tl.load(sin_t_ptr + pos_t * pairs_t + pair_t, mask=valid & is_t, other=0.0)
    cos_h = tl.load(cos_h_ptr + pos_h * pairs_h + pair_h, mask=valid & is_h, other=0.0)
    sin_h = tl.load(sin_h_ptr + pos_h * pairs_h + pair_h, mask=valid & is_h, other=0.0)
    cos_w = tl.load(cos_w_ptr + pos_w * pairs_w + pair_w, mask=valid & is_w, other=0.0)
    sin_w = tl.load(sin_w_ptr + pos_w * pairs_w + pair_w, mask=valid & is_w, other=0.0)
    cos = cos_t + cos_h + cos_w
    sin = sin_t + sin_h + sin_w

    even_offsets = rows * (pairs_per_head * 2) + pair_in_head * 2
    odd_offsets = even_offsets + 1
    query_even = tl.load(query_ptr + even_offsets, mask=valid).to(tl.float32)
    query_odd = tl.load(query_ptr + odd_offsets, mask=valid).to(tl.float32)
    key_even = tl.load(key_ptr + even_offsets, mask=valid).to(tl.float32)
    key_odd = tl.load(key_ptr + odd_offsets, mask=valid).to(tl.float32)

    # Eager materializes ``query * scale`` in BF16 before the fp32 RoPE.
    query_even = round_bf16_to_fp32(mul_rn_f32(query_even, query_scale))
    query_odd = round_bf16_to_fp32(mul_rn_f32(query_odd, query_scale))

    query_even_cos = mul_rn_f32(query_even, cos)
    query_odd_sin = mul_rn_f32(query_odd, sin)
    query_even_sin = mul_rn_f32(query_even, sin)
    query_odd_cos = mul_rn_f32(query_odd, cos)
    key_even_cos = mul_rn_f32(key_even, cos)
    key_odd_sin = mul_rn_f32(key_odd, sin)
    key_even_sin = mul_rn_f32(key_even, sin)
    key_odd_cos = mul_rn_f32(key_odd, cos)

    tl.store(
        query_output_ptr + even_offsets,
        add_rn_f32(query_even_cos, -query_odd_sin),
        mask=valid,
    )
    tl.store(
        query_output_ptr + odd_offsets,
        add_rn_f32(query_even_sin, query_odd_cos),
        mask=valid,
    )
    tl.store(
        key_output_ptr + even_offsets,
        add_rn_f32(key_even_cos, -key_odd_sin),
        mask=valid,
    )
    tl.store(
        key_output_ptr + odd_offsets,
        add_rn_f32(key_even_sin, key_odd_cos),
        mask=valid,
    )


def _axis_tables(
    length: int,
    dim: int,
    device: torch.device,
    base: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    exponents = torch.arange(0, dim, 2, dtype=torch.float64, device=device) / dim
    inv_freqs = (1.0 / base**exponents).to(torch.float32)
    positions = torch.arange(length, dtype=torch.float32, device=device)
    angles = positions[:, None] * inv_freqs[None, :]
    return angles.cos().contiguous(), angles.sin().contiguous()


def get_rope_tables(
    query: torch.Tensor,
    dim_split: tuple[int, int, int],
    base: float,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    num_frames, height, width = query.shape[1:4]
    cache_key = (query.device, num_frames, height, width, dim_split, base)
    cached = _TABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    tables = tuple(
        _axis_tables(length, dim, query.device, base)
        for length, dim in zip((num_frames, height, width), dim_split, strict=True)
    )
    if len(_TABLE_CACHE) >= _TABLE_CACHE_MAX:
        _TABLE_CACHE.pop(next(iter(_TABLE_CACHE)))
    _TABLE_CACHE[cache_key] = tables
    return tables


def _rotate_axis_reference(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    axis: int,
) -> torch.Tensor:
    output_dtype = hidden_states.dtype
    pairs = hidden_states.reshape(*hidden_states.shape[:-1], hidden_states.shape[-1] // 2, 2)
    even = pairs[..., 0].float()
    odd = pairs[..., 1].float()
    shape = [1, 1, 1, 1, 1, cos.shape[1]]
    shape[axis] = cos.shape[0]
    cos = cos.reshape(shape)
    sin = sin.reshape(shape)
    rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1)
    return rotated.reshape(hidden_states.shape).to(output_dtype)


def _apply_reference(
    hidden_states: torch.Tensor,
    tables: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    dim_split: tuple[int, int, int],
) -> torch.Tensor:
    outputs = []
    offset = 0
    for axis, (dim, (cos, sin)) in enumerate(zip(dim_split, tables, strict=True), 1):
        outputs.append(
            _rotate_axis_reference(
                hidden_states[..., offset : offset + dim],
                cos,
                sin,
                axis,
            )
        )
        offset += dim
    return torch.cat(outputs, dim=-1)


def _reference(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: float,
    tables: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    dim_split: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        _apply_reference(query * scale, tables, dim_split),
        _apply_reference(key, tables, dim_split),
    )


def _supported_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: float,
    dim_split: tuple[int, int, int],
    base: float,
) -> bool:
    return (
        is_ltx2_ops_eligible(query)
        and query.device.index not in _FAILED_DEVICES
        and query.dtype is torch.bfloat16
        and key.dtype is query.dtype
        and key.is_cuda
        and key.device == query.device
        and query.ndim == 6
        and key.shape == query.shape
        and query.is_contiguous()
        and key.is_contiguous()
        and query.numel() > 0
        and query.shape[-1] == 64
        and len(dim_split) == 3
        and sum(dim_split) == query.shape[-1]
        and all(dim > 0 and dim % 2 == 0 for dim in dim_split)
        and math.isfinite(scale)
        and math.isfinite(base)
        and base > 0
    )


def _launch(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: float,
    tables: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    dim_split: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    query_output = torch.empty_like(query)
    key_output = torch.empty_like(key)
    num_pairs = query.numel() // 2
    cos_t, sin_t = tables[0]
    cos_h, sin_h = tables[1]
    cos_w, sin_w = tables[2]
    with torch.accelerator.device_index(query.device.index):
        _qk_scale_rope_3d_kernel[(triton.cdiv(num_pairs, _BLOCK_PAIRS),)](
            query_output,
            key_output,
            query,
            key,
            cos_t,
            sin_t,
            cos_h,
            sin_h,
            cos_w,
            sin_w,
            num_pairs,
            query.shape[1] * query.shape[2] * query.shape[3],
            query.shape[2],
            query.shape[3],
            query.shape[4],
            scale,
            pairs_per_head=query.shape[5] // 2,
            pairs_t=dim_split[0] // 2,
            pairs_h=dim_split[1] // 2,
            block_pairs=_BLOCK_PAIRS,
            num_warps=8,
        )
    return query_output, key_output


def try_qk_scale_rope_3d_exact(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: float,
    dim_split: tuple[int, int, int],
    base: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run the verified-CUDA Triton fusion, or expose the eager expression."""

    if not _supported_inputs(query, key, scale, dim_split, base):
        return None
    device_index = query.device.index
    verification_contract = (device_index, dim_split, float(scale), float(base))
    try:
        tables = get_rope_tables(query, dim_split, base)
        output = _launch(query, key, scale, tables, dim_split)
        if verification_contract not in _VERIFIED_CONTRACTS:
            reference = _reference(query, key, scale, tables, dim_split)
            if not all(torch.equal(actual, expected) for actual, expected in zip(output, reference, strict=True)):
                _FAILED_DEVICES.add(device_index)
                logger.warning(
                    "Disabling LTX-2.5 DiffVAE Triton Q/K scale+3D RoPE on %s after a bit-exactness mismatch",
                    query.device,
                )
                return None
            _VERIFIED_CONTRACTS.add(verification_contract)
    except Exception as exc:  # noqa: BLE001 - fail closed after optimized-path failure
        _FAILED_DEVICES.add(device_index)
        logger.warning(
            "Disabling LTX-2.5 DiffVAE Triton Q/K scale+3D RoPE on %s after failure: %s",
            query.device,
            exc,
        )
        return None
    return output


__all__ = ["try_qk_scale_rope_3d_exact"]
