# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright 2026 Lightricks Ltd.
#
# Architecture and decode semantics ported and modified from Lightricks/LTX-2
# commit a95ab85. This file is the vLLM-Omni integration of that native
# implementation; Diffusers is used only for generic model/config interfaces.

"""Native vLLM-Omni implementation of the LTX-2.5 diffusion VAE decoder."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_kernels_available, logging
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.utils.constants import DIFFUSERS_DISABLE_REMOTE_CODE
from diffusers.utils.torch_utils import randn_tensor
from safetensors import safe_open

from .ltx2_diffusion_decoder_keyframes import (
    KeyframeStream,
    LTX2DecodeKeyframes,
    keyframe_clip_times,
    planes_for_tile,
    remaining_time_strides,
    upsample_keyframe_planes,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

LTX25_NATIVE_DIFFUSION_DECODER_REPO_ID = "Lightricks/LTX-2.5"
LTX25_NATIVE_ARTIFACT_REVISION = "8a4ff96f581e72bedc1b44367581c49d544a05f1"
LTX25_NATIVE_DIFFUSION_DECODER_FILENAME = "vae/ltx-2.5-video-vae-bf16.safetensors"
LTX25_NATIVE_SOURCE_REVISION = "a95ab856bf29407b6b066ede0abe1846050db56c"

_NATIVE_DECODER_PREFIXES = ("vae.decoder.", "decoder.")
_NATIVE_STATISTICS_KEYS = {
    "per_channel_statistics.mean-of-means": "latents_mean",
    "per_channel_statistics.std-of-means": "latents_std",
}
_NATIVE_KEY_REPLACEMENTS = (
    ("t_embedder.mlp.0.", "t_embedder.timestep_embedder.linear_1."),
    ("t_embedder.mlp.2.", "t_embedder.timestep_embedder.linear_2."),
    (".attn.proj.", ".attn.to_out.0."),
    (".attn.q_norm.", ".attn.norm_q."),
    (".attn.k_norm.", ".attn.norm_k."),
)
_GATE_FOLD_TARGETS = {
    ".attn.to_out.0.weight": ".gate_msa",
    ".attn.to_out.0.bias": ".gate_msa",
    ".mlp.w_down.weight": ".gate_mlp",
    ".context_proj.weight": ".gate_ctx",
    ".context_proj.bias": ".gate_ctx",
}
_GATE_SUFFIXES = tuple(_GATE_FOLD_TARGETS.values())


def convert_ltx25_native_diffusion_decoder_config(native_config: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the official safetensors VAE metadata into the Omni wrapper config."""
    vae_config = native_config.get("vae", native_config)
    if not isinstance(vae_config, Mapping):
        raise ValueError("Native LTX-2.5 VAE config must contain a mapping named `vae`.")
    decoder_config = vae_config.get("decoder", vae_config)
    if not isinstance(decoder_config, Mapping):
        raise ValueError("Native LTX-2.5 VAE config must contain a decoder mapping.")

    stage_channels = tuple(decoder_config["stage_channels"])
    stage_depths = tuple(decoder_config["stage_depths"])
    stage_kernels = tuple(tuple(kernel) for kernel in decoder_config["stage_kernels"])
    upsamples = tuple((tuple(stride), int(reduction)) for stride, reduction in decoder_config["upsamples"])
    if len(stage_channels) != 5 or len(stage_depths) != 5 or len(stage_kernels) < 4 or len(upsamples) != 4:
        raise ValueError(
            "Native LTX-2.5 DiffVAE must define five stages and four upsample hops; got "
            f"channels={len(stage_channels)}, depths={len(stage_depths)}, kernels={len(stage_kernels)}, "
            f"upsamples={len(upsamples)}."
        )

    patch_size = int(decoder_config.get("patch_size", 4))
    spatial_compression_ratio = patch_size * math.prod(stride[1] for stride, _ in upsamples)
    temporal_compression_ratio = math.prod(stride[0] for stride, _ in upsamples)
    model_output_type = decoder_config.get("model_output_type", vae_config.get("model_output_type", "v"))
    stage5_kernel = tuple(decoder_config.get("stage5_kernel", stage_kernels[-1]))
    return {
        "out_channels": int(decoder_config.get("out_channels", 3)),
        "latent_channels": int(decoder_config.get("in_channels", 128)),
        "patch_size": patch_size,
        "scaling_factor": 1.0,
        "decoder_head_dim": int(decoder_config.get("head_dim", 64)),
        "decoder_stage_channels": stage_channels,
        "decoder_stage_depths": stage_depths,
        "decoder_stage_kernels": stage_kernels[:4],
        "decoder_upsample_strides": tuple(stride for stride, _ in upsamples),
        "decoder_upsample_channel_reductions": tuple(reduction for _, reduction in upsamples),
        "decoder_stage5_kernel": stage5_kernel,
        "decoder_t_emb_dim": int(decoder_config.get("t_emb_dim", 384)),
        "decoder_timestep_scale_multiplier": float(decoder_config.get("timestep_scale_multiplier", 1.0)),
        "decoder_model_output_type": str(model_output_type),
        "decoder_num_inference_steps": int(decoder_config.get("default_num_inference_steps", 2)),
        "spatial_compression_ratio": spatial_compression_ratio,
        "temporal_compression_ratio": temporal_compression_ratio,
    }


def load_ltx25_native_diffusion_decoder_config(path: str) -> dict[str, Any]:
    """Read decoder architecture from the official checkpoint metadata."""
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    raw_config = metadata.get("config")
    if raw_config is None:
        raise ValueError(f"Native LTX-2.5 DiffVAE checkpoint {path!r} has no `config` metadata.")
    try:
        parsed = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Native LTX-2.5 DiffVAE checkpoint {path!r} has invalid JSON config metadata.") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"Native LTX-2.5 DiffVAE checkpoint {path!r} config metadata must be an object.")
    return convert_ltx25_native_diffusion_decoder_config(parsed)


def _strip_native_decoder_prefix(key: str) -> str | None:
    for prefix in _NATIVE_DECODER_PREFIXES:
        if key.startswith(prefix):
            return key.removeprefix(prefix)
    return None


def _native_statistics_target(key: str) -> str | None:
    return _NATIVE_STATISTICS_KEYS.get(key.removeprefix("vae."))


def _rename_native_decoder_key(key: str) -> str:
    for source, target in _NATIVE_KEY_REPLACEMENTS:
        key = key.replace(source, target)
    return key


def _fold_native_gate(key: str, value: torch.Tensor, gates: Mapping[str, torch.Tensor]) -> torch.Tensor:
    target_suffix = next((suffix for suffix in _GATE_FOLD_TARGETS if key.endswith(suffix)), None)
    if target_suffix is None:
        return value
    gate_key = key[: -len(target_suffix)] + _GATE_FOLD_TARGETS[target_suffix]
    gate = gates.get(gate_key)
    if gate is None:
        return value
    gate = gate.to(device=value.device, dtype=torch.float32)
    value_float = value.to(dtype=torch.float32)
    folded = gate.unsqueeze(1) * value_float if value.ndim == 2 else gate * value_float
    return folded.to(dtype=value.dtype)


def convert_ltx25_native_diffusion_decoder_state_dict(
    native_state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert the canonical checkpoint into this runtime's local parameter layout."""
    decoder_state_dict: dict[str, torch.Tensor] = {}
    converted: dict[str, torch.Tensor] = {}
    for key, value in native_state_dict.items():
        statistics_target = _native_statistics_target(key)
        if statistics_target is not None:
            converted[statistics_target] = value
            continue
        decoder_key = _strip_native_decoder_prefix(key)
        if decoder_key is not None:
            decoder_state_dict[decoder_key] = value

    gates = {key: value for key, value in decoder_state_dict.items() if key.endswith(_GATE_SUFFIXES)}
    for key, value in decoder_state_dict.items():
        # Preview heads and legacy static gates are not runtime parameters. The
        # static gates are folded into their associated projection below.
        if key.startswith("coarse_") or ".coarse_" in key or key in gates:
            continue

        converted_key = _rename_native_decoder_key(key)
        value = _fold_native_gate(converted_key, value, gates)
        if converted_key.endswith((".qkv.weight", ".qkv.bias")):
            leaf = "weight" if converted_key.endswith(".weight") else "bias"
            prefix = converted_key[: -len(f"qkv.{leaf}")]
            if value.shape[0] % 3 != 0:
                raise ValueError(
                    f"Fused LTX-2.5 DiffVAE parameter {key!r} has leading dimension "
                    f"{value.shape[0]}, which is not divisible by 3."
                )
            chunk = value.shape[0] // 3
            converted[f"decoder.{prefix}to_q.{leaf}"] = value[:chunk].clone()
            converted[f"decoder.{prefix}to_k.{leaf}"] = value[chunk : 2 * chunk].clone()
            converted[f"decoder.{prefix}to_v.{leaf}"] = value[2 * chunk :].clone()
            continue

        converted[f"decoder.{converted_key}"] = value

    return converted


def load_ltx25_native_diffusion_decoder_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Read and convert only DiffVAE tensors from the canonical Native VAE file."""
    with safe_open(path, framework="pt", device="cpu") as handle:
        native_state_dict = {
            key: handle.get_tensor(key)
            for key in handle.keys()
            if _strip_native_decoder_prefix(key) is not None or _native_statistics_target(key) is not None
        }
    return convert_ltx25_native_diffusion_decoder_state_dict(native_state_dict)


def _patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Space-to-depth on H/W only: `(B, C, F, H, W)` -> `(B, C * patch_size**2, F, H // p, W // p)`.

    The channel packing order is `(channel, width_offset, height_offset)`, matching the reference implementation's `b c
    (f p) (h q) (w r) -> b (c p r q) f h w` with `p = 1`.
    """
    batch_size, num_channels, num_frames, height, width = x.shape
    x = x.reshape(
        batch_size, num_channels, num_frames, height // patch_size, patch_size, width // patch_size, patch_size
    )
    x = x.permute(0, 1, 6, 4, 2, 3, 5)
    return x.reshape(
        batch_size, num_channels * patch_size * patch_size, num_frames, height // patch_size, width // patch_size
    )


def _unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Depth-to-space on H/W only, the exact inverse of [`_patchify`]."""
    batch_size, num_channels, num_frames, height, width = x.shape
    num_channels = num_channels // (patch_size * patch_size)
    x = x.reshape(batch_size, num_channels, patch_size, patch_size, num_frames, height, width)
    x = x.permute(0, 1, 4, 5, 3, 6, 2)
    return x.reshape(batch_size, num_channels, num_frames, height * patch_size, width * patch_size)


class LTX2VideoVaeRotaryPosEmbed3D(nn.Module):
    """Absolute 3D rotary embedding for the diffusion decoder's neighborhood attention.

    `head_dim` is split into (T, H, W) chunks, each rotated by its own axis position. Positions are the tensor's own
    0-based indices: attention here is always a local window with no causal masking, so the score between a query and a
    key depends only on their relative offset and a shared origin shift is a no-op. Rotation is computed in fp32 and
    cast back to the input dtype.
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 8 != 0:
            raise ValueError(f"head_dim must be a multiple of 8, got {head_dim}.")
        # Split `head_dim` across the (T, H, W) chunks the way the reference decoder does: a quarter to T, the
        # rest halved between H and W, with both halves kept even so each holds whole rotation pairs.
        dim_t = (head_dim // 4) // 2 * 2
        dim_hw = (head_dim - dim_t) // 2
        if dim_hw % 2 != 0:
            dim_t -= 2
            dim_hw = (head_dim - dim_t) // 2
        if dim_t <= 0 or dim_hw <= 0:
            raise ValueError(f"head_dim must leave positive T/H/W RoPE chunks, got {head_dim}.")
        self.rope_dim_split = (dim_t, dim_hw, dim_hw)
        self.base = base

    def _inv_freqs(self, dim: int, device: torch.device) -> torch.Tensor:
        # Match the native LTX implementation exactly: form the frequency
        # table in NumPy float64, cast once to float32, then move it.
        exponents = np.arange(0, dim, 2, dtype=np.float64) / dim
        inv_freqs = 1.0 / np.power(float(self.base), exponents)
        return torch.from_numpy(inv_freqs).to(device=device, dtype=torch.float32)

    def _rotate_axis(self, x: torch.Tensor, positions: torch.Tensor, inv_freqs: torch.Tensor, axis: int):
        out_dtype = x.dtype
        pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        even = pairs[..., 0].float()
        odd = pairs[..., 1].float()
        # Broadcast the angle over (B, T, H, W, heads, dim // 2), varying only along `axis`.
        shape = [1, 1, 1, 1, 1, inv_freqs.shape[0]]
        shape[axis] = positions.shape[0]
        angles = (positions[:, None] * inv_freqs[None, :]).reshape(shape)
        cos, sin = angles.cos(), angles.sin()
        rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1)
        return rotated.reshape(x.shape).to(out_dtype)

    def forward(self, hidden_states: torch.Tensor, times: torch.Tensor | None = None) -> torch.Tensor:
        """`hidden_states`: `(B, T, H, W, heads, head_dim)`."""
        dim_t, dim_h, _ = self.rope_dim_split
        num_frames, height, width = hidden_states.shape[1:4]
        device = hidden_states.device
        inv_t, inv_h, inv_w = (self._inv_freqs(dim, device) for dim in self.rope_dim_split)

        positions_t = (
            torch.arange(num_frames, dtype=torch.float32, device=device)
            if times is None
            else times.to(device=device, dtype=torch.float32)
        )
        if positions_t.shape != (num_frames,):
            raise ValueError(f"Temporal RoPE positions must have shape ({num_frames},), got {positions_t.shape}.")
        positions_h = torch.arange(height, dtype=torch.float32, device=device)
        positions_w = torch.arange(width, dtype=torch.float32, device=device)
        rotated_t = self._rotate_axis(hidden_states[..., :dim_t], positions_t, inv_t, axis=1)
        rotated_h = self._rotate_axis(hidden_states[..., dim_t : dim_t + dim_h], positions_h, inv_h, axis=2)
        rotated_w = self._rotate_axis(hidden_states[..., dim_t + dim_h :], positions_w, inv_w, axis=3)
        return torch.cat([rotated_t, rotated_h, rotated_w], dim=-1)


class _LTX2VideoVaeNattenRequiredProcessor:
    """Construction-only default replaced by NATTEN during pipeline setup."""

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self, attn: LTX2VideoVaeNeighborhoodAttention, hidden_states: torch.Tensor, block_mask=None
    ) -> torch.Tensor:
        del attn, hidden_states, block_mask
        raise RuntimeError(
            "LTX-2.5 DiffVAE requires its NATTEN attention processor. "
            "Construct it through the LTX pipeline component loader."
        )


class LTX2VideoVaeNeighborhoodNattenProcessor:
    """Neighborhood-attention processor using NATTEN's `na3d`, which is what the reference decoder calls.

    NATTEN is fetched from the Hub (`shi-labs/natten`, a trusted kernel publisher) through the `kernels` package rather
    than imported from a local install, so it needs `kernels`, a supported GPU, and `DIFFUSERS_DISABLE_REMOTE_CODE`
    unset; `backend=None` lets NATTEN pick the fastest kernel for the device. There is no runtime CPU fallback; the
    pinned official LTX source is the reference implementation.

    `na3d` encodes the neighborhood window in the kernel itself, so this processor takes no `block_mask` and the
    decoder never builds one for it.
    """

    def __init__(self, backend: str | None = None):
        if DIFFUSERS_DISABLE_REMOTE_CODE:
            raise ValueError(
                "LTX2VideoVaeNeighborhoodNattenProcessor downloads the `shi-labs/natten` kernel from the Hub, which "
                "is disabled globally by the `DIFFUSERS_DISABLE_REMOTE_CODE` environment variable. Unset it to run "
                "the LTX-2.5 DiffVAE decoder."
            )
        if not is_kernels_available():
            raise ImportError(
                "LTX2VideoVaeNeighborhoodNattenProcessor fetches NATTEN from the Hub with the `kernels` package. "
                "Install it with `pip install kernels` to run the LTX-2.5 DiffVAE decoder."
            )
        from kernels import get_kernel

        self._na3d = get_kernel("shi-labs/natten", version=1).na3d
        self.backend = backend

    def __call__(
        self, attn: LTX2VideoVaeNeighborhoodAttention, hidden_states: torch.Tensor, block_mask=None
    ) -> torch.Tensor:
        batch_size, num_frames, height, width, channels = hidden_states.shape
        query, key, value = attn.project_qkv(hidden_states)
        # NATTEN's CUTLASS kernels silently produce wrong output for non-contiguous inputs.
        query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        # `scale=1.0`: the query is already scaled in `project_qkv`, as in the reference.
        hidden_states = self._na3d(query, key, value, kernel_size=attn.kernel_size, scale=1.0, backend=self.backend)
        hidden_states = hidden_states.reshape(batch_size, num_frames, height, width, channels)
        return attn.to_out[0](hidden_states)


class LTX2VideoVaeNeighborhoodAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = _LTX2VideoVaeNattenRequiredProcessor
    _available_processors = [_LTX2VideoVaeNattenRequiredProcessor, LTX2VideoVaeNeighborhoodNattenProcessor]
    # The processor reads `to_q`/`to_k`/`to_v` directly and has no fused path, so QKV fusion would build an unused
    # `to_qkv` and silently no-op. Pipeline setup installs NATTEN through `set_attn_processor` instead.
    _supports_qkv_fusion = False

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        head_dim: int = 64,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if dim % head_dim != 0:
            raise ValueError(f"dim {dim} must be divisible by head_dim {head_dim}.")
        self.heads = dim // head_dim
        self.head_dim = head_dim
        self.kernel_size = tuple(kernel_size)
        self.scale = head_dim**-0.5

        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=True), nn.Dropout(0.0)])
        self.norm_q = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(head_dim, eps=1e-6)
        self.rope = LTX2VideoVaeRotaryPosEmbed3D(head_dim, base=rope_base)
        self.set_processor(self._default_processor_cls())

    def project_qkv(
        self,
        hidden_states: torch.Tensor,
        times: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Q/K/V as `(B, T, H, W, heads, head_dim)`, RMS-normed, query pre-scaled, then rotated.

        The query carries the `1 / sqrt(head_dim)` factor here so both processors can ask their attention backend for
        `scale=1.0` — this is the order the reference uses (norm, scale, then rotate).
        """
        batch_size, num_frames, height, width, _ = hidden_states.shape
        shape = (batch_size, num_frames, height, width, self.heads, self.head_dim)
        query = self.to_q(hidden_states).view(shape)
        key = self.to_k(hidden_states).view(shape)
        value = self.to_v(hidden_states).view(shape)

        query = self.norm_q(query)
        key = self.norm_k(key)
        query = query * self.scale
        return self.rope(query, times), self.rope(key, times), value

    def forward_with_keyframes(
        self,
        hidden_states: torch.Tensor,
        keyframes: KeyframeStream,
    ) -> tuple[torch.Tensor, KeyframeStream]:
        """Joint local attention for a video volume and independent keyframe planes."""
        if hidden_states.device.type != "cuda":
            raise RuntimeError(
                "LTX-2.5 keyframe-aware DiffVAE decode requires CUDA for its joint Triton attention kernel."
            )
        batch_size, num_frames, height, width, _ = hidden_states.shape
        planes = keyframes.x.shape[1]
        query, key, value = self.project_qkv(hidden_states)
        keyframe_query, keyframe_key, keyframe_value = self.project_qkv(keyframes.x, keyframes.times)
        from .ltx2_diffusion_decoder_joint_triton import joint_na3d

        attended, keyframe_attended = joint_na3d(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            keyframe_query.contiguous(),
            keyframe_key.contiguous(),
            keyframe_value.contiguous(),
            keyframes.times,
            keyframes.valid,
            self.kernel_size,
        )
        attended = self.to_out[0](attended.reshape(batch_size, num_frames, height, width, -1))
        keyframe_attended = self.to_out[0](keyframe_attended.reshape(batch_size, planes, height, width, -1))
        return attended, replace(keyframes, x=keyframe_attended)

    def build_block_mask(self, hidden_states: torch.Tensor):
        """Native NATTEN/eager processors encode their window directly."""
        del hidden_states
        return None

    def forward(self, hidden_states: torch.Tensor, block_mask=None) -> torch.Tensor:
        """Channels-last in and out: `(B, T, H, W, C)`."""
        num_frames, height, width = hidden_states.shape[1:4]
        kernel_t, kernel_h, kernel_w = self.kernel_size
        if num_frames < kernel_t or height < kernel_h or width < kernel_w:
            raise ValueError(
                f"Neighborhood attention requires each spatial dim to be at least its kernel size; got "
                f"(T, H, W) = ({num_frames}, {height}, {width}) with kernel_size {self.kernel_size}."
            )
        return self.processor(self, hidden_states, block_mask)


# Tokens per tile in `LTX2VideoVaeSwiGLU`, matching the reference decoder's own default. `w_gate(x)` and
# `w_up(x)` are both hidden-width and their product makes a third, so evaluating a whole video at once
# holds three hidden-width tensors live at the same time — at 121 frames and 512x768 that is
# 3 x 5.67 GiB, which by itself dominates decode memory. Fixing a token *count* rather than a number of
# tiles keeps that bound independent of resolution.
_SWIGLU_TILE_SIZE = 16384


class LTX2VideoVaeSwiGLU(nn.Module):
    """Gated MLP: `w_down(silu(w_gate(x)) * w_up(x))`, evaluated in tiles of `_SWIGLU_TILE_SIZE` tokens.

    Tiling is not an approximation: the MLP is pointwise across tokens, so splitting it changes only how many
    hidden-width elements exist at once, never what is computed. Outputs can still differ from the untiled evaluation
    by a few ulps, since a matmul over a tile may reduce in a different order than the full-tensor call.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, *token_dims, channels = hidden_states.shape
        num_tokens = math.prod(token_dims)
        if num_tokens <= _SWIGLU_TILE_SIZE:
            return self.w_down(F.silu(self.w_gate(hidden_states)) * self.w_up(hidden_states))

        flat = hidden_states.reshape(batch_size, num_tokens, channels)
        out = torch.empty_like(flat)
        for start in range(0, num_tokens, _SWIGLU_TILE_SIZE):
            tile = flat[:, start : start + _SWIGLU_TILE_SIZE]
            out[:, start : start + _SWIGLU_TILE_SIZE] = self.w_down(F.silu(self.w_gate(tile)) * self.w_up(tile))
        return out.reshape(hidden_states.shape)


def _swiglu_hidden_dim(dim: int, mlp_ratio: float) -> int:
    return (int(dim * mlp_ratio) + 15) // 16 * 16


class LTX2VideoVaeNABlock(nn.Module):
    """Pre-norm neighborhood-attention block used by the deterministic upsampling stages."""

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = LTX2VideoVaeNeighborhoodAttention(dim, kernel_size, head_dim=head_dim)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = LTX2VideoVaeSwiGLU(dim, _swiglu_hidden_dim(dim, mlp_ratio))

    def forward(self, hidden_states: torch.Tensor, block_mask=None) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), block_mask)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

    def forward_with_keyframes(
        self,
        hidden_states: torch.Tensor,
        keyframes: KeyframeStream,
    ) -> tuple[torch.Tensor, KeyframeStream]:
        video_attn, keyframe_attn = self.attn.forward_with_keyframes(
            self.norm1(hidden_states),
            replace(keyframes, x=self.norm1(keyframes.x)),
        )
        hidden_states = hidden_states + video_attn
        keyframe_x = keyframes.x + keyframe_attn.x
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        keyframe_x = keyframe_x + self.mlp(self.norm2(keyframe_x))
        return hidden_states, replace(keyframes, x=keyframe_x)


class LTX2VideoVaeAdaLNZero(nn.Module):
    """Shared AdaLN-Zero modulation: a timestep embedding to seven `(B, 1, 1, 1, C)` chunks.

    Seven chunks (scale/shift/gate for attention and MLP, plus a context gate) is the reference's shape. Only the four
    scale/shift chunks are consumed: the decoder's residuals are ungated, and the static gates the checkpoint used to
    carry are folded into the following linear weights at conversion time.
    """

    def __init__(self, dim: int, t_emb_dim: int, num_chunks: int = 7):
        super().__init__()
        self.num_chunks = num_chunks
        self.proj = nn.Linear(t_emb_dim, num_chunks * dim, bias=True)

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        chunks = self.proj(F.silu(t_emb)).chunk(self.num_chunks, dim=-1)
        return tuple(chunk[:, None, None, None, :] for chunk in chunks)


class LTX2VideoVaeDiffusionNABlock(nn.Module):
    """Neighborhood attention + SwiGLU, modulated by the shared AdaLN-Zero scale/shift.

    The decoder owns one `LTX2VideoVaeAdaLNZero`; each block adds its own `scale_shift_table` residual on top of it,
    injects the latent context through `context_proj`, and keeps its residuals ungated.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        context_channels: int,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        num_mod_params: int = 7,
    ):
        super().__init__()
        self.context_channels = context_channels
        self.num_mod_params = num_mod_params
        self.context_proj = nn.Linear(context_channels, dim, bias=True)
        self.scale_shift_table = nn.Parameter(torch.zeros(num_mod_params, dim))

        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = LTX2VideoVaeNeighborhoodAttention(dim, kernel_size, head_dim=head_dim)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = LTX2VideoVaeSwiGLU(dim, _swiglu_hidden_dim(dim, mlp_ratio))

    def forward(
        self,
        hidden_states: torch.Tensor,
        latent_context: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        block_mask=None,
    ) -> torch.Tensor:
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [
            modulation[i] + self.scale_shift_table[i].view(1, 1, 1, 1, -1) for i in range(self.num_mod_params)
        ]

        hidden_states = hidden_states + self.context_proj(latent_context)
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states) * (1 + scale_msa) + shift_msa, block_mask)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp)
        return hidden_states

    def forward_with_keyframes(
        self,
        hidden_states: torch.Tensor,
        keyframes: KeyframeStream,
        latent_context: torch.Tensor,
        keyframe_context: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, KeyframeStream]:
        """Stage-5 block with shared weights and one joint attention softmax."""
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [
            modulation[i] + self.scale_shift_table[i].view(1, 1, 1, 1, -1) for i in range(self.num_mod_params)
        ]
        hidden_states = hidden_states + self.context_proj(latent_context)
        keyframe_x = keyframes.x + self.context_proj(keyframe_context)
        video_attn, keyframe_attn = self.attn.forward_with_keyframes(
            self.norm1(hidden_states) * (1 + scale_msa) + shift_msa,
            replace(
                keyframes,
                x=self.norm1(keyframe_x) * (1 + scale_msa) + shift_msa,
            ),
        )
        hidden_states = hidden_states + video_attn
        keyframe_x = keyframe_x + keyframe_attn.x
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp)
        keyframe_x = keyframe_x + self.mlp(self.norm2(keyframe_x) * (1 + scale_mlp) + shift_mlp)
        # Official stage 5 re-applies the validity mask after every block.  This
        # matters for padded plane slots: context injection has a bias, so an
        # invalid zero plane would otherwise become non-zero before the next
        # joint-attention block.
        keyframe_x = keyframe_x * keyframes.valid[None, :, None, None, None]
        return hidden_states, replace(keyframes, x=keyframe_x)


class LTX2VideoVaePixelShuffleUpsampler(nn.Module):
    """Linear channel expansion followed by a channels-last pixel shuffle.

    When the temporal stride is 2 the shuffle produces a duplicate leading frame, which is dropped to keep the causal
    1:2 (composed 1:8) frame mapping. `drop_leading_frame=False` keeps it: a tiled decode passes that for temporal
    tiles that do not contain t=0, where the first input frame is an interior frame whose two output frames are both
    real content.
    """

    def __init__(self, in_channels: int, stride: tuple[int, int, int], out_channels_reduction_factor: int = 1):
        super().__init__()
        self.stride = tuple(stride)
        proj_out_channels = math.prod(self.stride) * in_channels // out_channels_reduction_factor
        self.out_channels = proj_out_channels // math.prod(self.stride)
        self.proj = nn.Linear(in_channels, proj_out_channels, bias=True)

    def forward(self, hidden_states: torch.Tensor, drop_leading_frame: bool = True) -> torch.Tensor:
        batch_size, num_frames, height, width, _ = hidden_states.shape
        stride_t, stride_h, stride_w = self.stride
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size, num_frames, height, width, self.out_channels, stride_t, stride_h, stride_w
        )
        # (b, t, p1, h, p2, w, p3, c) -> merge each stride into its own axis
        hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 3, 7, 4)
        hidden_states = hidden_states.reshape(
            batch_size, num_frames * stride_t, height * stride_h, width * stride_w, self.out_channels
        )
        if stride_t == 2 and drop_leading_frame:
            hidden_states = hidden_states[:, 1:]
        return hidden_states


class LTX2VideoDiffusionDecoder3d(nn.Module):
    """The LTX-2.5 diffusion video decoder.

    Stages 1-4 deterministically upsample the latent into a context volume with neighborhood-attention blocks. Stage 5
    then denoises patchified pixels, conditioned on that context through AdaLN-Zero scale/shift. With
    `model_output_type="x0"` and a single step — how LTX-2.5 ships — stage 5 runs once and its prediction *is* the
    output; more steps add reverse Euler updates.
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        patch_size: int = 4,
        head_dim: int = 64,
        stage_channels: tuple[int, ...] = (2048, 1024, 512, 512, 256),
        stage_depths: tuple[int, ...] = (4, 6, 4, 2, 8),
        stage_kernels: tuple[tuple[int, int, int], ...] = ((3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5)),
        upsample_strides: tuple[tuple[int, int, int], ...] = ((1, 2, 2), (2, 1, 1), (2, 2, 2), (2, 2, 2)),
        upsample_channel_reductions: tuple[int, ...] = (2, 2, 1, 2),
        stage5_kernel: tuple[int, int, int] = (11, 11, 11),
        t_emb_dim: int = 384,
        temporal_compression_ratio: int = 8,
        timestep_scale_multiplier: float = 1000.0,
        model_output_type: str = "x0",
        default_num_inference_steps: int = 1,
    ):
        super().__init__()
        if model_output_type not in ("x0", "v"):
            raise ValueError(f"model_output_type must be 'x0' or 'v', got {model_output_type!r}.")
        # Each upsample divides the channel count by its reduction factor, so the stage widths and the
        # reductions are two views of the same thing and an inconsistent pair would only fail deep inside
        # the first block.
        for stage_idx, reduction in enumerate(upsample_channel_reductions):
            expected = stage_channels[stage_idx] // reduction
            if stage_channels[stage_idx + 1] != expected:
                raise ValueError(
                    f"stage_channels[{stage_idx + 1}] must be stage_channels[{stage_idx}] // "
                    f"upsample_channel_reductions[{stage_idx}] = {expected}, got {stage_channels[stage_idx + 1]}."
                )

        self.patch_size = patch_size
        self.out_channels = out_channels
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.model_output_type = model_output_type
        self.default_num_inference_steps = default_num_inference_steps
        self.temporal_compression_ratio = temporal_compression_ratio
        self.context_channels = stage_channels[-1]
        self.stage5_kernel = tuple(stage5_kernel)
        # NATTEN shifts its window inward at the grid border, so the last latent frame is replicated
        # through stages 1-4 and cropped off the context before stage 5, moving that border past the
        # frames that are kept.
        self.trailing_pad_latent_frames = (stage_kernels[0][0] // 2) * 2

        self.conv_in = nn.Linear(in_channels, stage_channels[0], bias=True)
        # DFR keyframe planes share every decoder weight with the video stream.
        # This input-space tag is the sole keyframe-specific learned parameter.
        self.type_emb = nn.Parameter(torch.zeros(in_channels))

        self.det_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for stage_idx, stride in enumerate(upsample_strides):
            channels = stage_channels[stage_idx]
            self.det_stages.append(
                nn.ModuleList(
                    [
                        LTX2VideoVaeNABlock(
                            dim=channels,
                            kernel_size=stage_kernels[stage_idx],
                            head_dim=head_dim,
                        )
                        for _ in range(stage_depths[stage_idx])
                    ]
                )
            )
            self.upsamples.append(
                LTX2VideoVaePixelShuffleUpsampler(
                    in_channels=channels,
                    stride=stride,
                    out_channels_reduction_factor=upsample_channel_reductions[stage_idx],
                )
            )

        self.t_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim=t_emb_dim, size_emb_dim=0)

        stage5_channels = stage_channels[-1]
        noised_pixel_channels = out_channels * patch_size**2
        self.conv_in_x_t = nn.Linear(noised_pixel_channels, stage5_channels, bias=True)
        self.shared_adaln = LTX2VideoVaeAdaLNZero(dim=stage5_channels, t_emb_dim=t_emb_dim)
        self.diff_blocks = nn.ModuleList(
            [
                LTX2VideoVaeDiffusionNABlock(
                    dim=stage5_channels,
                    kernel_size=stage5_kernel,
                    context_channels=self.context_channels,
                    head_dim=head_dim,
                    num_mod_params=self.shared_adaln.num_chunks,
                )
                for _ in range(stage_depths[-1])
            ]
        )
        self.norm_out = nn.RMSNorm(stage5_channels, eps=1e-6)
        self.conv_out = nn.Linear(stage5_channels, noised_pixel_channels, bias=True)
        self._keyframe_time_strides = remaining_time_strides(self.upsamples)

    def _keyframe_stream_from_latents(
        self,
        keyframes: LTX2DecodeKeyframes,
        valid: torch.Tensor | None = None,
    ) -> KeyframeStream:
        keyframe_x = keyframes.latents.permute(0, 2, 3, 4, 1)
        keyframe_x = self.conv_in(keyframe_x + self.type_emb.view(1, 1, 1, 1, -1))
        if valid is None:
            valid = torch.ones(keyframe_x.shape[1], dtype=torch.bool, device=keyframe_x.device)
        times = keyframe_clip_times(
            keyframes.pixel_frame_indices,
            self._keyframe_time_strides[0],
            keyframes.clip_start_frame,
        )
        return KeyframeStream(
            x=keyframe_x,
            times=times.to(device=keyframe_x.device),
            valid=valid.to(device=keyframe_x.device),
        ).masked()

    def _run_det_stage_with_keyframes(
        self,
        hidden_states: torch.Tensor,
        keyframes: KeyframeStream,
        stage_index: int,
        *,
        drop_leading_frame: bool,
        pixel_frame_indices: torch.Tensor,
        next_time_origin: float = 0.0,
        clip_start_frame: int = 0,
    ) -> tuple[torch.Tensor, KeyframeStream]:
        for block in self.det_stages[stage_index]:
            hidden_states, keyframes = block.forward_with_keyframes(hidden_states, keyframes)
        hidden_states = self.upsamples[stage_index](
            hidden_states,
            drop_leading_frame=drop_leading_frame,
        )
        keyframe_x = upsample_keyframe_planes(self.upsamples[stage_index], keyframes.x)
        times = keyframe_clip_times(
            pixel_frame_indices,
            self._keyframe_time_strides[stage_index + 1],
            clip_start_frame,
            extra_origin=next_time_origin,
        )
        return hidden_states, KeyframeStream(
            x=keyframe_x,
            times=times.to(device=keyframe_x.device),
            valid=keyframes.valid,
        ).masked()

    def forward_stages_1_to_3_with_keyframes(
        self,
        hidden_states: torch.Tensor,
        keyframes: LTX2DecodeKeyframes,
        *,
        keyframe_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KeyframeStream]:
        """Run the shared deterministic stages over video and keyframe streams."""
        keyframes.validate()
        num_pad = self.trailing_pad_latent_frames
        if num_pad > 0:
            hidden_states = torch.cat(
                [hidden_states, hidden_states[:, :, -1:].expand(-1, -1, num_pad, -1, -1)],
                dim=2,
            )
        video = self.conv_in(hidden_states.permute(0, 2, 3, 4, 1))
        stream = self._keyframe_stream_from_latents(keyframes, keyframe_valid)
        for stage_index in range(len(self.det_stages) - 1):
            video, stream = self._run_det_stage_with_keyframes(
                video,
                stream,
                stage_index,
                drop_leading_frame=True,
                pixel_frame_indices=keyframes.pixel_frame_indices,
                clip_start_frame=keyframes.clip_start_frame,
            )
        return video, stream

    def forward_stage_4_with_keyframes(
        self,
        hidden_states: torch.Tensor,
        keyframes: KeyframeStream,
        pixel_frame_indices: torch.Tensor,
        *,
        drop_leading_frame: bool = True,
        crop_trailing_ghost: bool = True,
        stage4_time_origin: float = 0.0,
        pixel_time_origin: float = 0.0,
        clip_start_frame: int = 0,
    ) -> tuple[torch.Tensor, KeyframeStream]:
        keyframes = replace(
            keyframes,
            times=keyframe_clip_times(
                pixel_frame_indices,
                self._keyframe_time_strides[-2],
                clip_start_frame,
                extra_origin=stage4_time_origin,
            ).to(device=keyframes.x.device),
        )
        hidden_states, keyframes = self._run_det_stage_with_keyframes(
            hidden_states,
            keyframes,
            len(self.det_stages) - 1,
            drop_leading_frame=drop_leading_frame,
            pixel_frame_indices=pixel_frame_indices,
            next_time_origin=pixel_time_origin,
            clip_start_frame=clip_start_frame,
        )
        if crop_trailing_ghost and self.trailing_pad_latent_frames > 0:
            content_frames = max(
                hidden_states.shape[1] - self.trailing_pad_latent_frames * self.temporal_compression_ratio,
                1,
            )
            keep_frames = min(hidden_states.shape[1], max(content_frames, self.stage5_kernel[0]))
            hidden_states = hidden_states[:, :keep_frames]
        return hidden_states, keyframes

    def forward_stages_1_to_3(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """All deterministic stages but the last: latent `(B, C, T, H, W)` to a channels-last feature volume.

        The trailing ghost frames added for NATTEN's border shift stay in the output; [`forward_stage_4`] crops them.
        The split at this point exists for tiled decoding: these stages are cheap enough to run on the full volume,
        while stage 4 and the diffusion stage — where the grid and the channel-hidden products get large — run per
        tile.
        """
        num_pad = self.trailing_pad_latent_frames
        if num_pad > 0:
            trailing = hidden_states[:, :, -1:].expand(-1, -1, num_pad, -1, -1)
            hidden_states = torch.cat([hidden_states, trailing], dim=2)

        hidden_states = hidden_states.permute(0, 2, 3, 4, 1)
        hidden_states = self.conv_in(hidden_states)
        for blocks, upsample in zip(self.det_stages[:-1], self.upsamples[:-1]):
            # The grid and kernel are fixed within a stage, so every block shares one mask.
            block_mask = blocks[0].attn.build_block_mask(hidden_states)
            for block in blocks:
                hidden_states = block(hidden_states, block_mask)
            hidden_states = upsample(hidden_states)
        return hidden_states

    def forward_stage_4(
        self, hidden_states: torch.Tensor, drop_leading_frame: bool = True, crop_trailing_ghost: bool = True
    ) -> torch.Tensor:
        """Last deterministic stage: [`forward_stages_1_to_3`] output to context `(B, T_5, H_5, W_5, C_5)`.

        The defaults describe the untiled decode. A tiled decode overrides them per temporal tile: only the tile
        containing t=0 drops the upsample's duplicate leading frame, and only the tile containing the video end carries
        the trailing ghost frames to crop.
        """
        blocks = self.det_stages[-1]
        block_mask = blocks[0].attn.build_block_mask(hidden_states)
        for block in blocks:
            hidden_states = block(hidden_states, block_mask)
        hidden_states = self.upsamples[-1](hidden_states, drop_leading_frame=drop_leading_frame)

        num_pad = self.trailing_pad_latent_frames
        if crop_trailing_ghost and num_pad > 0:
            content_frames = max(hidden_states.shape[1] - num_pad * self.temporal_compression_ratio, 1)
            keep_frames = min(hidden_states.shape[1], max(content_frames, self.stage5_kernel[0]))
            hidden_states = hidden_states[:, :keep_frames]
        return hidden_states

    def forward_diffusion_step(
        self, latent_context: torch.Tensor, x_t: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """One stage-5 step. Returns the model's prediction in pixel space, `(B, C, F, H, W)`."""
        t_emb = self.t_embedder(
            self.timestep_scale_multiplier * timestep,
            resolution=None,
            aspect_ratio=None,
            batch_size=timestep.shape[0],
            hidden_dtype=latent_context.dtype,
        )
        modulation = self.shared_adaln(t_emb)

        hidden_states = _patchify(x_t, self.patch_size).permute(0, 2, 3, 4, 1)
        hidden_states = self.conv_in_x_t(hidden_states)
        block_mask = self.diff_blocks[0].attn.build_block_mask(hidden_states)
        for block in self.diff_blocks:
            hidden_states = block(hidden_states, latent_context, modulation, block_mask)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        hidden_states = hidden_states.permute(0, 4, 1, 2, 3).contiguous()
        return _unpatchify(hidden_states, self.patch_size)

    def forward_diffusion_step_with_keyframes(
        self,
        latent_context: torch.Tensor,
        keyframe_context: KeyframeStream,
        x_t: torch.Tensor,
        keyframe_x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One stage-5 step over the video and keyframe pixel-noise streams."""
        t_emb = self.t_embedder(
            self.timestep_scale_multiplier * timestep,
            resolution=None,
            aspect_ratio=None,
            batch_size=timestep.shape[0],
            hidden_dtype=latent_context.dtype,
        )
        modulation = self.shared_adaln(t_emb)
        hidden_states = self.conv_in_x_t(_patchify(x_t, self.patch_size).permute(0, 2, 3, 4, 1))
        keyframe_x = self.conv_in_x_t(_patchify(keyframe_x_t, self.patch_size).permute(0, 2, 3, 4, 1))
        keyframe_x = keyframe_x * keyframe_context.valid[None, :, None, None, None]
        keyframe_stream = replace(keyframe_context, x=keyframe_x)
        for block in self.diff_blocks:
            hidden_states, keyframe_stream = block.forward_with_keyframes(
                hidden_states,
                keyframe_stream,
                latent_context,
                keyframe_context.x,
                modulation,
            )

        def output_pixels(value: torch.Tensor) -> torch.Tensor:
            value = self.conv_out(self.norm_out(value))
            value = value.permute(0, 4, 1, 2, 3).contiguous()
            return _unpatchify(value, self.patch_size)

        return output_pixels(hidden_states), output_pixels(keyframe_stream.x)

    def denoise(self, latent_context: torch.Tensor, x_t: torch.Tensor, num_inference_steps: int) -> torch.Tensor:
        """Denoise `x_t` `(B, C, F, H, W)` through the stage-5 diffusion loop, conditioned on `latent_context`."""
        batch_size = latent_context.shape[0]
        timesteps = torch.linspace(
            1.0, 1.0 / num_inference_steps, num_inference_steps, device=latent_context.device, dtype=torch.float32
        )

        if num_inference_steps == 1 and self.model_output_type == "x0":
            return self.forward_diffusion_step(latent_context, x_t, timesteps[:1].expand(batch_size))

        for step_idx in range(num_inference_steps):
            t_now = timesteps[step_idx].expand(batch_size)
            t_next = timesteps[step_idx + 1] if step_idx + 1 < num_inference_steps else torch.zeros_like(t_now)
            model_out = self.forward_diffusion_step(latent_context, x_t, t_now).float()
            x_t_fp32 = x_t.float()
            if self.model_output_type == "x0":
                sigma = t_now.view(-1, *([1] * (x_t.ndim - 1)))
                model_out = (x_t_fp32 - model_out) / sigma
            dt = (t_now - t_next).view(-1, *([1] * (x_t.ndim - 1)))
            x_t = (x_t_fp32 - dt * model_out).to(x_t.dtype)
        return x_t

    def denoise_with_keyframes(
        self,
        latent_context: torch.Tensor,
        keyframe_context: KeyframeStream,
        x_t: torch.Tensor,
        keyframe_x_t: torch.Tensor,
        num_inference_steps: int,
    ) -> torch.Tensor:
        """Evolve both pixel streams and return only the decoded video stream."""
        batch_size = latent_context.shape[0]
        timesteps = torch.linspace(
            1.0,
            1.0 / num_inference_steps,
            num_inference_steps,
            device=latent_context.device,
            dtype=torch.float32,
        )
        for step_index in range(num_inference_steps):
            t_now = timesteps[step_index].expand(batch_size)
            t_next = (
                timesteps[step_index + 1].expand(batch_size)
                if step_index + 1 < num_inference_steps
                else torch.zeros_like(t_now)
            )
            model_out, keyframe_model_out = self.forward_diffusion_step_with_keyframes(
                latent_context,
                keyframe_context,
                x_t,
                keyframe_x_t,
                t_now,
            )

            def euler(sample: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
                sample_fp32 = sample.float()
                prediction = prediction.float()
                if self.model_output_type == "x0":
                    sigma = t_now.view(-1, *([1] * (sample.ndim - 1)))
                    prediction = (sample_fp32 - prediction) / sigma
                dt = (t_now - t_next).view(-1, *([1] * (sample.ndim - 1)))
                return (sample_fp32 - dt * prediction).to(sample.dtype)

            x_t = euler(x_t, model_out)
            keyframe_x_t = euler(keyframe_x_t, keyframe_model_out)
        return x_t

    def decode_tile_with_keyframes(
        self,
        features: torch.Tensor,
        keyframe_features: KeyframeStream,
        pixel_frame_indices: torch.Tensor,
        *,
        drop_leading_frame: bool,
        crop_trailing_ghost: bool,
        stage4_time_origin: float,
        pixel_time_origin: float,
        clip_start_frame: int,
        generator: torch.Generator | list[torch.Generator] | None,
        num_inference_steps: int,
        x_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent_context, keyframe_context = self.forward_stage_4_with_keyframes(
            features,
            keyframe_features,
            pixel_frame_indices,
            drop_leading_frame=drop_leading_frame,
            crop_trailing_ghost=crop_trailing_ghost,
            stage4_time_origin=stage4_time_origin,
            pixel_time_origin=pixel_time_origin,
            clip_start_frame=clip_start_frame,
        )
        pixel_shape = (
            latent_context.shape[0],
            self.out_channels,
            latent_context.shape[1],
            latent_context.shape[2] * self.patch_size,
            latent_context.shape[3] * self.patch_size,
        )
        if x_t is None:
            x_t = randn_tensor(
                pixel_shape,
                generator=generator,
                device=latent_context.device,
                dtype=latent_context.dtype,
            )
        keyframe_x_t = randn_tensor(
            (
                pixel_shape[0],
                pixel_shape[1],
                keyframe_context.x.shape[1],
                pixel_shape[3],
                pixel_shape[4],
            ),
            generator=generator,
            device=latent_context.device,
            dtype=latent_context.dtype,
        )
        return self.denoise_with_keyframes(
            latent_context,
            keyframe_context,
            x_t,
            keyframe_x_t,
            num_inference_steps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        num_inference_steps = num_inference_steps or self.default_num_inference_steps
        latent_context = self.forward_stage_4(self.forward_stages_1_to_3(hidden_states))
        # The context grid is the stage-5 token grid, so the pixel canvas is its shape times the patch size —
        # temporally that is the causal (T - 1) * ratio + 1 mapping of the LTX-2 latent space.
        pixel_shape = (
            hidden_states.shape[0],
            self.out_channels,
            latent_context.shape[1],
            latent_context.shape[2] * self.patch_size,
            latent_context.shape[3] * self.patch_size,
        )
        x_t = randn_tensor(pixel_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype)
        return self.denoise(latent_context, x_t, num_inference_steps)

    def forward_with_keyframes(
        self,
        hidden_states: torch.Tensor,
        keyframes: LTX2DecodeKeyframes,
        generator: torch.Generator | list[torch.Generator] | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        num_inference_steps = num_inference_steps or self.default_num_inference_steps
        features, keyframe_features = self.forward_stages_1_to_3_with_keyframes(hidden_states, keyframes)
        return self.decode_tile_with_keyframes(
            features,
            keyframe_features,
            keyframes.pixel_frame_indices,
            drop_leading_frame=True,
            crop_trailing_ghost=True,
            stage4_time_origin=0.0,
            pixel_time_origin=0.0,
            clip_start_frame=keyframes.clip_start_frame,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )


def _tile_intervals(length: int, tile_size: int, stride: int, min_size: int) -> list[tuple[int, int]]:
    """Overlapping `[start, end)` tiles covering `[0, length)`, with starts spaced `stride` apart.

    A trailing remnant shorter than `min_size` is merged into the previous tile instead of decoded on its own:
    neighborhood attention rejects any grid smaller than its kernel, so a remnant tile cannot always stand alone.
    """
    if length <= tile_size:
        return [(0, length)]
    starts = list(range(0, length, stride))
    while len(starts) > 1 and length - starts[-1] < min_size:
        starts.pop()
    return [(start, min(start + tile_size, length)) for start in starts[:-1]] + [(starts[-1], length)]


class LTX2VideoDiffusionDecoderModel(ModelMixin, AttentionMixin, ConfigMixin):
    r"""
    The LTX-2 diffusion video decoder, introduced in LTX-2.5.

    This is a decoder, not an autoencoder: it has no encoder and cannot produce latents. Encoding stays with
    [`AutoencoderKLLTX2Video`], whose latent space this consumes unchanged, so latents are interchangeable between the
    convolutional decoder and this one.

    It is also a diffusion model rather than a deterministic decoder — it denoises pixels conditioned on a context
    volume built from the latents — which is why it is driven by [`LTX2VideoDiffusionDecodePipeline`] rather than being
    passed as a pipeline's `vae`.

    The latent statistics are carried here as buffers so the decode pipeline can denormalize without loading a second
    autoencoder just for two vectors.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    # Both block types close over a residual add that combines outputs from different children, so a device split
    # inside either one would separate tensors that have to meet again in the same forward.
    _no_split_modules = ["LTX2VideoVaeNABlock", "LTX2VideoVaeDiffusionNABlock"]
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 128,
        patch_size: int = 4,
        scaling_factor: float = 1.0,
        decoder_head_dim: int = 64,
        decoder_stage_channels: tuple[int, ...] = (2048, 1024, 512, 512, 256),
        decoder_stage_depths: tuple[int, ...] = (4, 6, 4, 2, 8),
        decoder_stage_kernels: tuple[tuple[int, int, int], ...] = ((3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5)),
        decoder_upsample_strides: tuple[tuple[int, int, int], ...] = ((1, 2, 2), (2, 1, 1), (2, 2, 2), (2, 2, 2)),
        decoder_upsample_channel_reductions: tuple[int, ...] = (2, 2, 1, 2),
        decoder_stage5_kernel: tuple[int, int, int] = (11, 11, 11),
        decoder_t_emb_dim: int = 384,
        decoder_timestep_scale_multiplier: float = 1000.0,
        decoder_model_output_type: str = "x0",
        decoder_num_inference_steps: int = 1,
        spatial_compression_ratio: int = 32,
        temporal_compression_ratio: int = 8,
    ) -> None:
        super().__init__()

        self.decoder = LTX2VideoDiffusionDecoder3d(
            in_channels=latent_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            head_dim=decoder_head_dim,
            stage_channels=decoder_stage_channels,
            stage_depths=decoder_stage_depths,
            stage_kernels=decoder_stage_kernels,
            upsample_strides=decoder_upsample_strides,
            upsample_channel_reductions=decoder_upsample_channel_reductions,
            stage5_kernel=decoder_stage5_kernel,
            t_emb_dim=decoder_t_emb_dim,
            temporal_compression_ratio=temporal_compression_ratio,
            timestep_scale_multiplier=decoder_timestep_scale_multiplier,
            model_output_type=decoder_model_output_type,
            default_num_inference_steps=decoder_num_inference_steps,
        )

        self.spatial_compression_ratio = spatial_compression_ratio
        self.temporal_compression_ratio = temporal_compression_ratio

        # When decoding a large enough video, the memory-dominant stages (the last deterministic stage and the
        # stage-5 diffusion blocks) can run on overlapping tiles that are blended back together. The earlier
        # stages always see the full latent, so tiling changes the output only near tile borders.
        self.use_tiling = False

        # The tile size and the distance between consecutive starts are in
        # decoded pixels/frames.  The 160px / 40-frame overlaps are the native
        # LTX receptive-field halos for the production stage-4/stage-5 layout.
        self.tile_sample_min_height = 768
        self.tile_sample_min_width = 768
        self.tile_sample_min_num_frames = 80
        self.tile_sample_stride_height = 608
        self.tile_sample_stride_width = 608
        self.tile_sample_stride_num_frames = 40

        latents_mean = torch.zeros((latent_channels,), requires_grad=False)
        latents_std = torch.ones((latent_channels,), requires_grad=False)
        self.register_buffer("latents_mean", latents_mean, persistent=True)
        self.register_buffer("latents_std", latents_std, persistent=True)

    @classmethod
    def from_ltx25_native_checkpoint(
        cls,
        checkpoint_path: str,
        config: Mapping[str, Any],
        dtype: torch.dtype,
    ) -> LTX2VideoDiffusionDecoderModel:
        """Construct the local decoder directly from the canonical Native checkpoint."""
        with torch.device("meta"):
            model = cls.from_config(dict(config))
        state_dict = load_ltx25_native_diffusion_decoder_state_dict(checkpoint_path)
        try:
            model.load_state_dict(state_dict, strict=True, assign=True)
        except RuntimeError as exc:
            raise ValueError(f"Invalid LTX-2.5 Native DiffVAE checkpoint {checkpoint_path!r}.") from exc
        model.to(device="cpu", dtype=dtype)
        return model

    def enable_tiling(
        self,
        tile_sample_min_height: int | None = None,
        tile_sample_min_width: int | None = None,
        tile_sample_min_num_frames: int | None = None,
        tile_sample_stride_height: int | None = None,
        tile_sample_stride_width: int | None = None,
        tile_sample_stride_num_frames: int | None = None,
    ) -> None:
        r"""
        Enable tiled decoding. The deterministic upsampling stages before the last one always process the full latent
        (they run at low resolution and are cheap); the last stage and the stage-5 diffusion blocks — which dominate
        decode memory — run on overlapping tiles whose seams are blended linearly.

        Args:
            tile_sample_min_height (`int`, *optional*):
                The height of one decoded tile, in pixels.
            tile_sample_min_width (`int`, *optional*):
                The width of one decoded tile, in pixels.
            tile_sample_min_num_frames (`int`, *optional*):
                The number of frames of one decoded tile.
            tile_sample_stride_height (`int`, *optional*):
                The distance in pixels between the tops of two consecutive vertical tiles; the difference to
                `tile_sample_min_height` is the blended overlap.
            tile_sample_stride_width (`int`, *optional*):
                The distance in pixels between the left edges of two consecutive horizontal tiles.
            tile_sample_stride_num_frames (`int`, *optional*):
                The distance in frames between the starts of two consecutive temporal tiles.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_num_frames = tile_sample_min_num_frames or self.tile_sample_min_num_frames
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width
        self.tile_sample_stride_num_frames = tile_sample_stride_num_frames or self.tile_sample_stride_num_frames

    def disable_tiling(self) -> None:
        r"""Disable tiled decoding, returning to decoding the whole video in one pass."""
        self.use_tiling = False

    # Copied from diffusers.models.autoencoders.autoencoder_kl_ltx2.AutoencoderKLLTX2Video.blend_v
    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    # Copied from diffusers.models.autoencoders.autoencoder_kl_ltx2.AutoencoderKLLTX2Video.blend_h
    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    # Copied from diffusers.models.autoencoders.autoencoder_kl_ltx2.AutoencoderKLLTX2Video.blend_t
    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (
                x / blend_extent
            )
        return b

    def tiled_decode(
        self,
        z: torch.Tensor,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
        keyframes: LTX2DecodeKeyframes | None = None,
    ) -> torch.Tensor:
        r"""Decode a batch of latents with the last deterministic stage and the diffusion stage running per tile.

        Tiles live on the grid entering the last deterministic stage, where one cell maps to a fixed block of output
        pixels; the `tile_sample_*` sizes are converted to that grid, so they should be multiples of the cell size (8
        px spatially and 2 frames temporally for the production config). Temporal tiles follow the causal frame
        mapping: the tile containing t=0 drops the temporal upsample's duplicate leading frame and only the tile
        containing the video end carries the NATTEN border padding.
        """
        if keyframes is not None:
            return self._tiled_decode_with_keyframes(
                z,
                keyframes,
                generator=generator,
                num_inference_steps=num_inference_steps,
            )

        decoder = self.decoder
        num_inference_steps = num_inference_steps or decoder.default_num_inference_steps
        batch_size = z.shape[0]
        patch_size = decoder.patch_size

        # Pixels per cell of the tiling grid: the last upsample's stride times the stage-5 patch size.
        upsample_stride = decoder.upsamples[-1].stride
        scale_t, scale_h, scale_w = (
            upsample_stride[0],
            upsample_stride[1] * patch_size,
            upsample_stride[2] * patch_size,
        )
        tile_t, stride_t = self.tile_sample_min_num_frames // scale_t, self.tile_sample_stride_num_frames // scale_t
        tile_h, stride_h = self.tile_sample_min_height // scale_h, self.tile_sample_stride_height // scale_h
        tile_w, stride_w = self.tile_sample_min_width // scale_w, self.tile_sample_stride_width // scale_w
        # Every tile must satisfy both remaining neighborhood-attention kernels: the last deterministic stage
        # sees the tile as-is, stage 5 sees it scaled by the upsample stride.
        min_sizes = [
            max(kernel_4, -(-kernel_5 // stride))
            for kernel_4, kernel_5, stride in zip(
                self.config.decoder_stage_kernels[-1], self.config.decoder_stage5_kernel, upsample_stride
            )
        ]

        features = decoder.forward_stages_1_to_3(z)
        # The trailing ghost frames replicate through the earlier stages' temporal upsamples, whose composed
        # mapping is affine with slope equal to the product of their strides.
        ghost_frames = decoder.trailing_pad_latent_frames * math.prod(up.stride[0] for up in decoder.upsamples[:-1])
        num_frames = features.shape[1] - ghost_frames
        height, width = features.shape[2], features.shape[3]

        temporal_tiles = _tile_intervals(num_frames, tile_t, stride_t, min_sizes[0])
        height_tiles = _tile_intervals(height, tile_h, stride_h, min_sizes[1])
        width_tiles = _tile_intervals(width, tile_w, stride_w, min_sizes[2])
        blend_frames = (tile_t - stride_t) * scale_t
        blend_height = (tile_h - stride_h) * scale_h
        blend_width = (tile_w - stride_w) * scale_w

        # A single-step x0 decode predicts pixels from pure noise, so each tile draws its own; a multi-step
        # decode integrates its noise across steps, so overlapping tiles must start from the same canvas.
        single_step_x0 = num_inference_steps == 1 and decoder.model_output_type == "x0"
        x_t_full = None
        if not single_step_x0:
            pixel_frames = num_frames * scale_t - (1 if scale_t == 2 else 0)
            x_t_full = randn_tensor(
                (batch_size, decoder.out_channels, pixel_frames, height * scale_h, width * scale_w),
                generator=generator,
                device=z.device,
                dtype=z.dtype,
            )

        frame_groups = []
        for t0, t1 in temporal_tiles:
            is_origin = t0 == 0
            is_trailing = t1 == num_frames
            # The tile containing the video end takes the ghost frames with it into stage 4.
            feature_t1 = features.shape[1] if is_trailing else t1
            rows = []
            for h0, h1 in height_tiles:
                row = []
                for w0, w1 in width_tiles:
                    context = decoder.forward_stage_4(
                        features[:, t0:feature_t1, h0:h1, w0:w1],
                        drop_leading_frame=is_origin,
                        crop_trailing_ghost=is_trailing,
                    )
                    tile_pixel_shape = (
                        batch_size,
                        decoder.out_channels,
                        context.shape[1],
                        context.shape[2] * patch_size,
                        context.shape[3] * patch_size,
                    )
                    if single_step_x0:
                        x_t = randn_tensor(tile_pixel_shape, generator=generator, device=z.device, dtype=z.dtype)
                    else:
                        # A non-origin tile keeps the duplicate leading frame, placing its first cell one pixel
                        # frame earlier than `t0 * scale_t` — the causal 1-then-`scale_t` frame mapping.
                        pixel_t0 = t0 * scale_t - (1 if not is_origin and scale_t == 2 else 0)
                        x_t = x_t_full[
                            :,
                            :,
                            pixel_t0 : pixel_t0 + tile_pixel_shape[2],
                            h0 * scale_h : h0 * scale_h + tile_pixel_shape[3],
                            w0 * scale_w : w0 * scale_w + tile_pixel_shape[4],
                        ]
                    row.append(decoder.denoise(context, x_t, num_inference_steps))
                rows.append(row)

            result_rows = []
            for i, row in enumerate(rows):
                result_row = []
                for j, tile in enumerate(row):
                    # blend the above tile and the left tile to the current tile and add the current tile to
                    # the result row
                    if i > 0:
                        tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                    if j > 0:
                        tile = self.blend_h(row[j - 1], tile, blend_width)
                    # The last tile can extend past the stride grid (a short remnant is merged into it), so it
                    # keeps its full extent instead of being cropped to the stride.
                    keep_height = stride_h * scale_h if i < len(rows) - 1 else tile.shape[3]
                    keep_width = stride_w * scale_w if j < len(row) - 1 else tile.shape[4]
                    result_row.append(tile[:, :, :, :keep_height, :keep_width])
                result_rows.append(torch.cat(result_row, dim=4))
            frame_groups.append(torch.cat(result_rows, dim=3))

        result = []
        for k, group in enumerate(frame_groups):
            if k > 0:
                group = self.blend_t(frame_groups[k - 1], group, blend_frames)
            if k < len(frame_groups) - 1:
                # The origin group is one frame short of `stride * scale`: its first cell decodes to a single
                # pixel frame under the causal mapping.
                keep_frames = stride_t * scale_t - (1 if k == 0 and scale_t == 2 else 0)
                group = group[:, :, :keep_frames]
            result.append(group)
        return torch.cat(result, dim=2)

    def _tiled_decode_with_keyframes(
        self,
        z: torch.Tensor,
        keyframes: LTX2DecodeKeyframes,
        *,
        generator: torch.Generator | list[torch.Generator] | None,
        num_inference_steps: int | None,
    ) -> torch.Tensor:
        """Keyframe-aware tiled decode using global plane times at every tile."""
        decoder = self.decoder
        num_inference_steps = num_inference_steps or decoder.default_num_inference_steps
        batch_size = z.shape[0]
        patch_size = decoder.patch_size
        upsample_stride = decoder.upsamples[-1].stride
        scale_t, scale_h, scale_w = (
            upsample_stride[0],
            upsample_stride[1] * patch_size,
            upsample_stride[2] * patch_size,
        )
        tile_t = self.tile_sample_min_num_frames // scale_t
        tile_h = self.tile_sample_min_height // scale_h
        tile_w = self.tile_sample_min_width // scale_w
        stride_t = self.tile_sample_stride_num_frames // scale_t
        stride_h = self.tile_sample_stride_height // scale_h
        stride_w = self.tile_sample_stride_width // scale_w
        min_sizes = [
            max(kernel_4, -(-kernel_5 // stride))
            for kernel_4, kernel_5, stride in zip(
                self.config.decoder_stage_kernels[-1],
                self.config.decoder_stage5_kernel,
                upsample_stride,
            )
        ]

        features, keyframe_features = decoder.forward_stages_1_to_3_with_keyframes(z, keyframes)
        ghost_frames = decoder.trailing_pad_latent_frames * math.prod(
            upsample.stride[0] for upsample in decoder.upsamples[:-1]
        )
        num_frames = features.shape[1] - ghost_frames
        height, width = features.shape[2:4]
        temporal_tiles = _tile_intervals(num_frames, tile_t, stride_t, min_sizes[0])
        height_tiles = _tile_intervals(height, tile_h, stride_h, min_sizes[1])
        width_tiles = _tile_intervals(width, tile_w, stride_w, min_sizes[2])
        blend_frames = (tile_t - stride_t) * scale_t
        blend_height = (tile_h - stride_h) * scale_h
        blend_width = (tile_w - stride_w) * scale_w

        single_step_x0 = num_inference_steps == 1 and decoder.model_output_type == "x0"
        x_t_full = None
        if not single_step_x0:
            pixel_frames = num_frames * scale_t - (1 if scale_t == 2 else 0)
            x_t_full = randn_tensor(
                (batch_size, decoder.out_channels, pixel_frames, height * scale_h, width * scale_w),
                generator=generator,
                device=z.device,
                dtype=z.dtype,
            )

        frame_groups = []
        for t0, t1 in temporal_tiles:
            is_origin = t0 == 0
            is_trailing = t1 == num_frames
            feature_t1 = features.shape[1] if is_trailing else t1
            pixel_t0 = t0 * scale_t - (1 if not is_origin and scale_t == 2 else 0)
            raw_pixel_frames = (feature_t1 - t0) * scale_t - (1 if is_origin and scale_t == 2 else 0)
            if is_trailing and decoder.trailing_pad_latent_frames > 0:
                content_frames = max(
                    raw_pixel_frames - decoder.trailing_pad_latent_frames * decoder.temporal_compression_ratio,
                    1,
                )
                raw_pixel_frames = min(raw_pixel_frames, max(content_frames, decoder.stage5_kernel[0]))
            plane_keep = planes_for_tile(
                keyframes.pixel_frame_indices,
                pixel_t0,
                pixel_t0 + raw_pixel_frames - 1,
                clip_start_frame=keyframes.clip_start_frame,
            )
            selected_indices = keyframes.pixel_frame_indices[plane_keep]
            selected_stream = keyframe_features.select_planes(plane_keep.to(keyframe_features.x.device))

            rows = []
            for h0, h1 in height_tiles:
                row = []
                for w0, w1 in width_tiles:
                    feature_tile = features[:, t0:feature_t1, h0:h1, w0:w1]
                    stream_tile = selected_stream.crop_spatial(slice(h0, h1), slice(w0, w1))
                    tile_pixel_shape = (
                        batch_size,
                        decoder.out_channels,
                        raw_pixel_frames,
                        (h1 - h0) * scale_h,
                        (w1 - w0) * scale_w,
                    )
                    if single_step_x0:
                        x_t = None
                    else:
                        x_t = x_t_full[
                            :,
                            :,
                            pixel_t0 : pixel_t0 + tile_pixel_shape[2],
                            h0 * scale_h : h0 * scale_h + tile_pixel_shape[3],
                            w0 * scale_w : w0 * scale_w + tile_pixel_shape[4],
                        ]
                    row.append(
                        decoder.decode_tile_with_keyframes(
                            feature_tile,
                            stream_tile,
                            selected_indices,
                            drop_leading_frame=is_origin,
                            crop_trailing_ghost=is_trailing,
                            stage4_time_origin=float(t0),
                            pixel_time_origin=float(pixel_t0),
                            clip_start_frame=keyframes.clip_start_frame,
                            generator=generator,
                            num_inference_steps=num_inference_steps,
                            x_t=x_t,
                        )
                    )
                rows.append(row)

            result_rows = []
            for row_index, row in enumerate(rows):
                result_row = []
                for column_index, tile in enumerate(row):
                    if row_index > 0:
                        tile = self.blend_v(rows[row_index - 1][column_index], tile, blend_height)
                    if column_index > 0:
                        tile = self.blend_h(row[column_index - 1], tile, blend_width)
                    keep_height = stride_h * scale_h if row_index < len(rows) - 1 else tile.shape[3]
                    keep_width = stride_w * scale_w if column_index < len(row) - 1 else tile.shape[4]
                    result_row.append(tile[:, :, :, :keep_height, :keep_width])
                result_rows.append(torch.cat(result_row, dim=4))
            frame_groups.append(torch.cat(result_rows, dim=3))

        result = []
        for time_index, group in enumerate(frame_groups):
            if time_index > 0:
                group = self.blend_t(frame_groups[time_index - 1], group, blend_frames)
            if time_index < len(frame_groups) - 1:
                keep_frames = stride_t * scale_t - (1 if time_index == 0 and scale_t == 2 else 0)
                group = group[:, :, :keep_frames]
            result.append(group)
        return torch.cat(result, dim=2)

    @apply_forward_hook
    def decode(
        self,
        z: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None = None,
        num_inference_steps: int | None = None,
        return_dict: bool = True,
        keyframes: LTX2DecodeKeyframes | None = None,
    ) -> DecoderOutput | torch.Tensor:
        """Decode a batch of latents.

        `z` is expected to be denormalized already (the pipeline applies `latents_mean` / `latents_std`), matching
        [`AutoencoderKLLTX2Video`]. This decoder denoises, so pass `generator` for reproducibility.
        """
        raw_keyframes = None
        if keyframes is not None:
            output_frames = (z.shape[2] - 1) * self.temporal_compression_ratio + 1
            keyframes.validate(num_frames=output_frames)
            if keyframes.latents.shape[:2] != z.shape[:2] or keyframes.latents.shape[-2:] != z.shape[-2:]:
                raise ValueError(
                    "LTX-2.5 decode keyframes must match video latent batch, channels, height, and width; "
                    f"got keyframes {tuple(keyframes.latents.shape)} and video {tuple(z.shape)}."
                )
            latents_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
            latents_std = self.latents_std.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
            raw_keyframes = replace(
                keyframes,
                latents=(
                    keyframes.latents.to(device=z.device, dtype=z.dtype) * latents_std / self.config.scaling_factor
                    + latents_mean
                ),
                pixel_frame_indices=keyframes.pixel_frame_indices.to(device=z.device),
            )

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_min_num_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio
        if self.use_tiling and (
            z.shape[2] > tile_latent_min_num_frames
            or z.shape[3] > tile_latent_min_height
            or z.shape[4] > tile_latent_min_width
        ):
            decoded = self.tiled_decode(
                z,
                generator=generator,
                num_inference_steps=num_inference_steps,
                keyframes=raw_keyframes,
            )
        elif raw_keyframes is not None:
            decoded = self.decoder.forward_with_keyframes(
                z,
                raw_keyframes,
                generator=generator,
                num_inference_steps=num_inference_steps,
            )
        else:
            decoded = self.decoder(z, generator=generator, num_inference_steps=num_inference_steps)

        # A short clip keeps replicated temporal context through stage 5 to
        # satisfy the neighborhood kernel. Do not expose those context pixels.
        target_num_frames = (z.shape[2] - 1) * self.temporal_compression_ratio + 1
        target_height = z.shape[3] * self.spatial_compression_ratio
        target_width = z.shape[4] * self.spatial_compression_ratio
        decoded = decoded[:, :, :target_num_frames, :target_height, :target_width]

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(
        self,
        z: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None = None,
        num_inference_steps: int | None = None,
        return_dict: bool = True,
        keyframes: LTX2DecodeKeyframes | None = None,
    ) -> DecoderOutput | tuple[torch.Tensor]:
        r"""
        Args:
            z (`torch.Tensor`):
                Latents of shape `(B, C, F, H, W)`, expected to be denormalized already (the pipeline applies
                `latents_mean` / `latents_std`), matching [`AutoencoderKLLTX2Video`].
            generator (`torch.Generator`, *optional*):
                This decoder denoises, so pass a generator to make decoding reproducible.
            num_inference_steps (`int`, *optional*):
                Number of denoising steps. Defaults to the decoder's `decoder_num_inference_steps` config value.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoders.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoders.vae.DecoderOutput`] or `tuple`
        """
        return self.decode(
            z,
            generator=generator,
            num_inference_steps=num_inference_steps,
            return_dict=return_dict,
            keyframes=keyframes,
        )
