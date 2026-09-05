# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exact LTX DiffVAE residual, RMSNorm, and AdaLN fusions."""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F
from vllm.triton_utils import tl, triton

from ..numerics import add_rn_f32, round_bf16_to_fp32
from ..platform import is_ltx2_ops_eligible

_HIDDEN_SIZE = 256
_POINTWISE_BLOCK = 1024
_VERIFY_ROWS = 256
_VERIFY_ELEMENTS = 1 << 20
_FAILED_ADALN_KEYS: set[tuple[int | None, int, float]] = set()
_VERIFIED_ADALN_KEYS: set[tuple[int | None, int, float]] = set()
_FAILED_ADD_DEVICES: set[int | None] = set()
_VERIFIED_ADD_DEVICES: set[int | None] = set()

logger = logging.getLogger(__name__)


@triton.jit
def _residual_add3_kernel(
    output_ptr,
    x_ptr,
    residual_a_ptr,
    residual_b_ptr,
    residual_c_ptr,
    elements,
    block: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    valid = offsets < elements
    value = add_rn_f32(
        tl.load(x_ptr + offsets, mask=valid, other=0.0).to(tl.float32),
        tl.load(residual_a_ptr + offsets, mask=valid, other=0.0).to(tl.float32),
    )
    value = round_bf16_to_fp32(value)
    value = add_rn_f32(
        value,
        tl.load(residual_b_ptr + offsets, mask=valid, other=0.0).to(tl.float32),
    )
    value = round_bf16_to_fp32(value)
    value = add_rn_f32(
        value,
        tl.load(residual_c_ptr + offsets, mask=valid, other=0.0).to(tl.float32),
    )
    tl.store(output_ptr + offsets, value, mask=valid)


def _same_activation(x: torch.Tensor, residual: torch.Tensor) -> bool:
    return (
        residual.dtype is torch.bfloat16
        and residual.is_cuda
        and residual.device == x.device
        and residual.shape == x.shape
        and residual.is_contiguous()
    )


def _modulation_matches(x: torch.Tensor, value: torch.Tensor) -> bool:
    return (
        value.dtype is torch.bfloat16
        and value.is_cuda
        and value.device == x.device
        and value.shape[-1] == _HIDDEN_SIZE
        and value.numel() == x.shape[0] * _HIDDEN_SIZE
        and value.is_contiguous()
    )


def _norm_weight_matches(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        weight.dtype is torch.bfloat16
        and weight.is_cuda
        and weight.device == x.device
        and weight.shape == (_HIDDEN_SIZE,)
        and weight.is_contiguous()
    )


def _adaln_inputs_supported(
    x: torch.Tensor,
    residual_a: torch.Tensor,
    residual_b: torch.Tensor | None,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> bool:
    return (
        is_ltx2_ops_eligible(x)
        and x.dtype is torch.bfloat16
        and x.ndim >= 2
        and x.shape[-1] == _HIDDEN_SIZE
        and x.numel() > 0
        and x.is_contiguous()
        and _same_activation(x, residual_a)
        and (residual_b is None or _same_activation(x, residual_b))
        and _norm_weight_matches(x, norm_weight)
        and _modulation_matches(x, scale)
        and _modulation_matches(x, shift)
        and math.isfinite(eps)
        and eps > 0
    )


def _residual_reference(
    x: torch.Tensor,
    residual_a: torch.Tensor,
    residual_b: torch.Tensor | None,
) -> torch.Tensor:
    hidden_states = x + residual_a
    if residual_b is not None:
        hidden_states = hidden_states + residual_b
    return hidden_states


def _adaln_reference(
    x: torch.Tensor,
    residual_a: torch.Tensor,
    residual_b: torch.Tensor | None,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    hidden_states = _residual_reference(x, residual_a, residual_b)
    normalized = F.rms_norm(hidden_states, (_HIDDEN_SIZE,), weight=norm_weight, eps=eps)
    return normalized * (1 + scale) + shift


def _launch_adaln(
    x: torch.Tensor,
    residual_a: torch.Tensor,
    residual_b: torch.Tensor | None,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    from .residual_adaln_tilelang import launch_residual_rms_norm_modulate_tilelang

    return launch_residual_rms_norm_modulate_tilelang(
        x,
        residual_a,
        residual_b,
        norm_weight,
        scale,
        shift,
        eps,
    )


def _verify_adaln_prefix(
    output: torch.Tensor,
    x: torch.Tensor,
    residual_a: torch.Tensor,
    residual_b: torch.Tensor | None,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> bool:
    rows_per_batch = x.numel() // (x.shape[0] * _HIDDEN_SIZE)
    rows = min(rows_per_batch, _VERIFY_ROWS)
    x_rows = x.reshape(x.shape[0], rows_per_batch, _HIDDEN_SIZE)[0, :rows]
    residual_a_rows = residual_a.reshape(x.shape[0], rows_per_batch, _HIDDEN_SIZE)[0, :rows]
    residual_b_rows = (
        None if residual_b is None else residual_b.reshape(x.shape[0], rows_per_batch, _HIDDEN_SIZE)[0, :rows]
    )
    scale_row = scale.reshape(x.shape[0], 1, _HIDDEN_SIZE)[0]
    shift_row = shift.reshape(x.shape[0], 1, _HIDDEN_SIZE)[0]
    reference = _adaln_reference(
        x_rows,
        residual_a_rows,
        residual_b_rows,
        norm_weight,
        scale_row,
        shift_row,
        eps,
    )
    actual = output.reshape(x.shape[0], rows_per_batch, _HIDDEN_SIZE)[0, :rows]
    return torch.equal(actual, reference)


def try_residual_rms_norm_modulate_exact(
    x: torch.Tensor,
    residual_a: torch.Tensor,
    residual_b: torch.Tensor | None,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    """Fuse one or two ordered BF16 residuals with RMSNorm and AdaLN."""

    if not _adaln_inputs_supported(x, residual_a, residual_b, norm_weight, scale, shift, eps):
        return None
    residual_count = 1 if residual_b is None else 2
    runtime_key = (x.device.index, residual_count, float(eps))
    if runtime_key in _FAILED_ADALN_KEYS:
        return None
    try:
        output = _launch_adaln(x, residual_a, residual_b, norm_weight, scale, shift, eps)
        if runtime_key not in _VERIFIED_ADALN_KEYS:
            if not _verify_adaln_prefix(
                output,
                x,
                residual_a,
                residual_b,
                norm_weight,
                scale,
                shift,
                eps,
            ):
                _FAILED_ADALN_KEYS.add(runtime_key)
                logger.warning(
                    "Disabling LTX DiffVAE residual-AdaLN fusion on %s after a bit-exactness mismatch",
                    x.device,
                )
                return None
            _VERIFIED_ADALN_KEYS.add(runtime_key)
    except Exception as exc:  # noqa: BLE001 - fail closed after optimized-path failure
        _FAILED_ADALN_KEYS.add(runtime_key)
        logger.warning(
            "Disabling LTX DiffVAE residual-AdaLN fusion on %s after failure: %s",
            x.device,
            exc,
        )
        return None
    return output


def _add_inputs_supported(x: torch.Tensor, residuals: tuple[torch.Tensor, ...]) -> bool:
    return (
        is_ltx2_ops_eligible(x)
        and x.dtype is torch.bfloat16
        and x.numel() > 0
        and x.is_contiguous()
        and all(_same_activation(x, residual) for residual in residuals)
    )


def _launch_add3(
    x: torch.Tensor,
    residual_a: torch.Tensor,
    residual_b: torch.Tensor,
    residual_c: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty_like(x)
    with torch.accelerator.device_index(x.device.index):
        _residual_add3_kernel[(triton.cdiv(x.numel(), _POINTWISE_BLOCK),)](
            output,
            x,
            residual_a,
            residual_b,
            residual_c,
            x.numel(),
            block=_POINTWISE_BLOCK,
            num_warps=4,
        )
    return output


def try_residual_add3_exact(
    x: torch.Tensor,
    residual_a: torch.Tensor,
    residual_b: torch.Tensor,
    residual_c: torch.Tensor,
) -> torch.Tensor | None:
    """Fuse three ordered BF16 residual additions into one pointwise pass."""

    residuals = (residual_a, residual_b, residual_c)
    if not _add_inputs_supported(x, residuals) or x.device.index in _FAILED_ADD_DEVICES:
        return None
    try:
        output = _launch_add3(x, *residuals)
        if x.device.index not in _VERIFIED_ADD_DEVICES:
            elements = min(x.numel(), _VERIFY_ELEMENTS)
            reference = (
                _residual_reference(
                    x.reshape(-1)[:elements],
                    residual_a.reshape(-1)[:elements],
                    residual_b.reshape(-1)[:elements],
                )
                + residual_c.reshape(-1)[:elements]
            )
            if not torch.equal(output.reshape(-1)[:elements], reference):
                _FAILED_ADD_DEVICES.add(x.device.index)
                logger.warning(
                    "Disabling LTX DiffVAE residual-add fusion on %s after a bit-exactness mismatch",
                    x.device,
                )
                return None
            _VERIFIED_ADD_DEVICES.add(x.device.index)
    except Exception as exc:  # noqa: BLE001 - fail closed after optimized-path failure
        _FAILED_ADD_DEVICES.add(x.device.index)
        logger.warning(
            "Disabling LTX DiffVAE residual-add fusion on %s after failure: %s",
            x.device,
            exc,
        )
        return None
    return output


__all__ = [
    "try_residual_add3_exact",
    "try_residual_rms_norm_modulate_exact",
]
