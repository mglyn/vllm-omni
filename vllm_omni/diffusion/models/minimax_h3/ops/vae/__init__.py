# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Install bit-exact eager optimizations for the MiniMax H3 video VAE."""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
import torch.nn as nn

from vllm_omni.diffusion.layers.activation import SiluAndMul

from .dispatch import resolve_h3_vae_operators


def _optimized_feed_forward(self: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return type(self).forward(self, hidden_states)

    hidden_states = self.w1(hidden_states)
    if self.use_gated:
        if hidden_states.is_cuda and hidden_states.dtype == torch.float16:
            hidden_states = self._omni_silu_and_mul(hidden_states)
        else:
            gate, hidden_states = hidden_states.chunk(2, dim=-1)
            hidden_states = self.act_fn(gate) * hidden_states
    else:
        hidden_states = self.act_fn(hidden_states)
    return self.w2(hidden_states)


def _optimized_attention(
    self: nn.Module,
    hidden_states: torch.Tensor,
    rotary_pos_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    pack_info: dict[str, Any] | None = None,
) -> torch.Tensor:
    pack_info = {} if pack_info is None else pack_info
    if torch.compiler.is_compiling() or self.spatial_parallel or rotary_pos_emb is None:
        return type(self).forward(self, hidden_states, rotary_pos_emb, pack_info)

    batch_size, sequence, _ = hidden_states.shape
    qkv = self.to_qkv(hidden_states).view(
        batch_size,
        sequence,
        -1,
        3 * self.dim_head,
    )
    query, key, value = qkv.chunk(3, dim=-1)
    optimized_qk = self._omni_qk_norm_rope(
        query,
        key,
        rotary_pos_emb,
        float(self.norm_q.eps),
    )
    if optimized_qk is None:
        return type(self).forward(self, hidden_states, rotary_pos_emb, pack_info)

    query, key = optimized_qk
    hidden_states = self.perform_attention(query, key, value, pack_info)
    hidden_states = hidden_states.reshape(batch_size, sequence, -1)
    return self.to_out(hidden_states)


def _optimized_transformer_block(
    self: nn.Module,
    hidden_states: torch.Tensor,
    rotary_pos_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    pack_info: dict[str, Any] | None = None,
) -> torch.Tensor:
    pack_info = {} if pack_info is None else pack_info
    if torch.compiler.is_compiling() or hidden_states.dtype != torch.float32:
        return type(self).forward(self, hidden_states, rotary_pos_emb, pack_info)

    normalized = self.norm1(hidden_states.float()).to(hidden_states.dtype)
    attention_output = self.attn(normalized, rotary_pos_emb, pack_info)
    updated = self._omni_scaled_residual(
        hidden_states,
        attention_output,
        self.scale1,
    )
    hidden_states = hidden_states + attention_output * self.scale1 if updated is None else updated

    normalized = self.norm2(hidden_states.float()).to(hidden_states.dtype)
    feed_forward_output = self.ff(normalized)
    updated = self._omni_scaled_residual(
        hidden_states,
        feed_forward_output,
        self.scale2,
    )
    return hidden_states + feed_forward_output * self.scale2 if updated is None else updated


def _decoder_block_linears(decoder: nn.Module) -> tuple[nn.Linear, ...] | None:
    blocks = getattr(decoder, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        return None

    linears: list[nn.Linear] = []
    for block in blocks:
        attention = getattr(block, "attn", None)
        feed_forward = getattr(block, "ff", None)
        candidates = (
            getattr(attention, "to_qkv", None),
            getattr(attention, "to_out", None),
            getattr(feed_forward, "w1", None),
            getattr(feed_forward, "w2", None),
        )
        if (
            not all(isinstance(linear, nn.Linear) for linear in candidates)
            or not isinstance(getattr(attention, "norm_q", None), nn.RMSNorm)
            or not isinstance(getattr(attention, "norm_k", None), nn.RMSNorm)
            or attention.norm_q.weight is not None
            or attention.norm_k.weight is not None
            or attention.norm_q.eps != attention.norm_k.eps
            or not isinstance(getattr(block, "norm1", None), nn.RMSNorm)
            or not isinstance(getattr(block, "norm2", None), nn.RMSNorm)
            or block.norm1.weight is None
            or block.norm2.weight is None
            or not getattr(block, "use_scale", False)
            or not getattr(feed_forward, "use_gated", False)
            or not isinstance(getattr(feed_forward, "act_fn", None), nn.SiLU)
        ):
            return None
        linears.extend(candidates)
    return tuple(linears)


def install_h3_vae_optimizations(
    decoder: nn.Module,
    *,
    device: torch.device,
) -> bool:
    """Install the operators selected for ``device`` once."""

    if getattr(decoder, "_omni_h3_vae_optimizations_installed", False):
        return True
    operators = resolve_h3_vae_operators(device)
    if operators is None:
        return False

    linears = _decoder_block_linears(decoder)
    if linears is None or any(linear.weight.dtype != torch.float32 for linear in linears):
        return False

    # The H3 decode path always uses FP16 CUDA autocast. Persisting these
    # rounded decoder-block weights avoids rebuilding the same casts per tile.
    for linear in linears:
        linear.to(dtype=torch.float16)

    for block in decoder.transformer_blocks:
        block.ff._omni_silu_and_mul = SiluAndMul()
        block.ff.forward = MethodType(_optimized_feed_forward, block.ff)
        block.attn._omni_qk_norm_rope = operators.qk_norm_rope
        block.attn.forward = MethodType(_optimized_attention, block.attn)
        block._omni_scaled_residual = operators.scaled_residual
        block.forward = MethodType(_optimized_transformer_block, block)

    decoder._omni_h3_vae_optimizations_installed = True
    return True


__all__ = ["install_h3_vae_optimizations"]
