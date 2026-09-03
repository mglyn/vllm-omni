# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Diffusers-module integration for optimized LTX DiffVAE eager operations."""

from __future__ import annotations

import logging
from functools import cache
from typing import Any

import torch
from diffusers.models.autoencoders.ltx2_diffusion_decoder import (
    LTX2VideoVaeDiffusionNABlock as DiffusersLTX2VideoVaeDiffusionNABlock,
)
from diffusers.models.autoencoders.ltx2_diffusion_decoder import (
    LTX2VideoVaeNeighborhoodAttention as DiffusersLTX2VideoVaeNeighborhoodAttention,
)
from diffusers.models.autoencoders.ltx2_diffusion_decoder import (
    LTX2VideoVaeNeighborhoodNattenProcessor as DiffusersLTX2VideoVaeNeighborhoodNattenProcessor,
)
from diffusers.models.autoencoders.ltx2_diffusion_decoder import (
    LTX2VideoVaeSwiGLU as DiffusersLTX2VideoVaeSwiGLU,
)
from torch import nn

from ..platform import is_ltx2_ops_eligible
from .qk_rms_norm import try_qk_rms_norm_scale_rope_3d_exact
from .residual_adaln import (
    try_residual_add3_exact,
    try_residual_rms_norm_modulate_exact,
)
from .swiglu import try_swiglu_tiled_exact

logger = logging.getLogger(__name__)


@cache
def _load_tilelang_fna():
    try:
        from .fna import try_fna3d_tilelang
    except (ImportError, OSError):
        return None
    return try_fna3d_tilelang


@cache
def _load_natten_token_permutation():
    from kernels import get_kernel

    natten = get_kernel("shi-labs/natten", version=1)
    return (
        natten.token_permute.token_permute_operation,
        natten.token_permute.token_unpermute_operation,
    )


class LTX2VideoVaeNeighborhoodNattenProcessor(DiffusersLTX2VideoVaeNeighborhoodNattenProcessor):
    """NATTEN processor with a bounded-drift SM90 stage-5 fast path."""

    def __call__(self, attn, hidden_states: torch.Tensor, block_mask=None) -> torch.Tensor:
        if tuple(attn.kernel_size) != (11, 11, 11) or not is_ltx2_ops_eligible(hidden_states):
            return super().__call__(attn, hidden_states, block_mask)

        fast_path = _load_tilelang_fna()
        if fast_path is None:
            return super().__call__(attn, hidden_states, block_mask)

        batch_size, num_frames, height, width, channels = hidden_states.shape
        query, key, value = attn.project_qkv(hidden_states)
        query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        try:
            token_permute, token_unpermute = _load_natten_token_permutation()
            query_permuted, token_shape, _ = token_permute(query, (4, 4, 4), flip_tiled_dims=True)
            key_permuted, _, _ = token_permute(key, (4, 4, 8), flip_tiled_dims=True)
            value_permuted, _, _ = token_permute(value, (4, 4, 8), flip_tiled_dims=True)
            optimized = fast_path(
                query_permuted,
                key_permuted,
                value_permuted,
                shape=(num_frames, height, width),
                window=tuple(attn.kernel_size),
            )
            if optimized is not None:
                hidden_states = token_unpermute(
                    optimized,
                    token_layout_shape=token_shape,
                    tile_shape=(4, 4, 4),
                    flip_tiled_dims=True,
                )
                hidden_states = hidden_states.reshape(batch_size, num_frames, height, width, channels)
                return attn.to_out[0](hidden_states)
        except Exception as exc:  # noqa: BLE001 - preserve the required NATTEN fallback
            logger.warning("Falling back to NATTEN after LTX DiffVAE TileLang FNA failure: %s", exc)

        hidden_states = self._na3d(
            query,
            key,
            value,
            kernel_size=attn.kernel_size,
            scale=1.0,
            backend=self.backend,
        )
        hidden_states = hidden_states.reshape(batch_size, num_frames, height, width, channels)
        return attn.to_out[0](hidden_states)


class LTX2VideoVaeNeighborhoodAttention(DiffusersLTX2VideoVaeNeighborhoodAttention):
    """DiffVAE attention with a fail-closed exact eager fast path."""

    def project_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_frames, height, width, _ = hidden_states.shape
        shape = (batch_size, num_frames, height, width, self.heads, self.head_dim)
        query = self.to_q(hidden_states).view(shape)
        key = self.to_k(hidden_states).view(shape)
        value = self.to_v(hidden_states).view(shape)

        optimized = try_qk_rms_norm_scale_rope_3d_exact(
            query,
            key,
            self.norm_q.weight,
            self.norm_k.weight,
            self.norm_q.eps,
            self.scale,
            self.rope.rope_dim_split,
            self.rope.base,
        )
        if optimized is not None:
            query, key = optimized
            return query, key, value

        query = self.norm_q(query)
        key = self.norm_k(key)
        query = query * self.scale
        return self.rope(query), self.rope(key), value


class LTX2VideoVaeSwiGLU(DiffusersLTX2VideoVaeSwiGLU):
    """DiffVAE SwiGLU with an exact workspace-reusing eager path."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        optimized = try_swiglu_tiled_exact(
            hidden_states,
            self.w_gate.weight,
            self.w_up.weight,
            self.w_down.weight,
        )
        if optimized is not None:
            return optimized
        return super().forward(hidden_states)


class LTX2VideoVaeDiffusionNABlock(DiffusersLTX2VideoVaeDiffusionNABlock):
    """DiffVAE block with exact eager residual-AdaLN fusions."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        latent_context: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        block_mask: Any = None,
    ) -> torch.Tensor:
        if hidden_states.dtype is not torch.bfloat16 or not is_ltx2_ops_eligible(hidden_states):
            return super().forward(hidden_states, latent_context, modulation, block_mask)

        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [
            modulation[i] + self.scale_shift_table[i].view(1, 1, 1, 1, -1) for i in range(self.num_mod_params)
        ]
        context_output = self.context_proj(latent_context)
        attention_input = try_residual_rms_norm_modulate_exact(
            hidden_states,
            context_output,
            None,
            self.norm1.weight,
            scale_msa,
            shift_msa,
            self.norm1.eps,
        )
        if attention_input is None:
            hidden_states = hidden_states + context_output
            hidden_states = hidden_states + self.attn(
                self.norm1(hidden_states) * (1 + scale_msa) + shift_msa,
                block_mask,
            )
            hidden_states = hidden_states + self.mlp(self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp)
            return hidden_states

        attention_output = self.attn(attention_input, block_mask)
        mlp_input = try_residual_rms_norm_modulate_exact(
            hidden_states,
            context_output,
            attention_output,
            self.norm2.weight,
            scale_mlp,
            shift_mlp,
            self.norm2.eps,
        )
        if mlp_input is None:
            residual = hidden_states + context_output
            residual = residual + attention_output
            mlp_input = self.norm2(residual) * (1 + scale_mlp) + shift_mlp
        mlp_output = self.mlp(mlp_input)
        output = try_residual_add3_exact(
            hidden_states,
            context_output,
            attention_output,
            mlp_output,
        )
        if output is not None:
            return output
        residual = hidden_states + context_output
        residual = residual + attention_output
        return residual + mlp_output


_OPTIMIZED_MODULE_TYPES: tuple[tuple[type[nn.Module], type[nn.Module]], ...] = (
    (DiffusersLTX2VideoVaeDiffusionNABlock, LTX2VideoVaeDiffusionNABlock),
    (
        DiffusersLTX2VideoVaeNeighborhoodAttention,
        LTX2VideoVaeNeighborhoodAttention,
    ),
    (DiffusersLTX2VideoVaeSwiGLU, LTX2VideoVaeSwiGLU),
)


def install_ltx2_diffvae_ops(decoder: nn.Module) -> None:
    """Install stateless optimized behavior on Diffusers-created modules."""

    for module in decoder.modules():
        for source_type, optimized_type in _OPTIMIZED_MODULE_TYPES:
            if isinstance(module, optimized_type):
                break
            if isinstance(module, source_type):
                module.__class__ = optimized_type
                break


__all__ = [
    "LTX2VideoVaeDiffusionNABlock",
    "LTX2VideoVaeNeighborhoodAttention",
    "LTX2VideoVaeNeighborhoodNattenProcessor",
    "LTX2VideoVaeSwiGLU",
    "install_ltx2_diffvae_ops",
]
