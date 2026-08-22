# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open

from vllm_omni.diffusion.lora_runtime import (
    DiffusionLoRABindingPlan,
    DiffusionLoRADeployment,
    DiffusionLoRASupport,
    LoadedDiffusionLoRA,
    LowRankUpdate,
    create_low_rank_executor,
)

_H3_TURBO_RANK = 128
_H3_TURBO_ALPHA = 128.0
_LORA_A_SUFFIX = ".lora_A.default.weight"
_LORA_B_SUFFIX = ".lora_B.default.weight"
_H3_LOGICAL_TARGETS = frozenset({"to_q", "to_k", "to_v", "out_proj", "fc1", "fc2"})

MINIMAX_H3_LORA_BINDING_PLAN = DiffusionLoRABindingPlan(
    component_names=("transformer",),
    target_modules=("to_q", "to_k", "to_v", "out_proj", "fc1", "fc2"),
    packed_modules_mapping={"qkv_proj": ("to_q", "to_k", "to_v")},
)


def _normalize_h3_target(raw_target: str) -> str:
    if raw_target.startswith("transformer_blocks."):
        target = "blocks." + raw_target.removeprefix("transformer_blocks.")
    elif raw_target.startswith("token_refiner.refiner_blocks."):
        target = "token_refiner.blocks." + raw_target.removeprefix("token_refiner.refiner_blocks.")
    else:
        raise ValueError(f"Unsupported MiniMax-H3 Turbo LoRA target prefix: {raw_target!r}")

    replacements = (
        (".attn.to_out.0", ".attn.out_proj"),
        (".ff.net.0.proj", ".mlp.fc1"),
        (".ff.net.2", ".mlp.fc2"),
    )
    for old, new in replacements:
        target = target.replace(old, new)
    leaf = target.rsplit(".", 1)[-1]
    if leaf not in _H3_LOGICAL_TARGETS:
        raise ValueError(f"Unsupported MiniMax-H3 Turbo LoRA logical target: {target!r}")
    return target


def _select_h3_turbo_file(artifact_path: Path) -> Path:
    if artifact_path.is_file():
        if artifact_path.suffix != ".safetensors":
            raise ValueError(f"MiniMax-H3 Turbo LoRA must be a safetensors file, got {artifact_path}")
        return artifact_path
    if not artifact_path.is_dir():
        raise ValueError(f"MiniMax-H3 Turbo LoRA artifact does not exist: {artifact_path}")

    candidates = sorted(artifact_path.glob("*v1.0*.safetensors"))
    if not candidates:
        candidates = sorted(artifact_path.glob("*.safetensors"))
    if len(candidates) != 1:
        raise ValueError(
            f"MiniMax-H3 Turbo LoRA artifact must resolve to exactly one v1.0 safetensors file, "
            f"found {[path.name for path in candidates]}"
        )
    return candidates[0]


class MiniMaxH3TurboLoRALoader:
    """Interpret the LightX2V v1.0 FL2VA Turbo release."""

    def __init__(self, pipeline: nn.Module) -> None:
        partition = getattr(pipeline, "partition", None)
        if partition == "ref2va":
            raise ValueError("MiniMax-H3 FL2VA Turbo LoRA is not supported by a Ref2VA-only service")

    def load(
        self,
        deployment: DiffusionLoRADeployment,
        artifact_path: Path,
    ) -> LoadedDiffusionLoRA:
        lora_file = _select_h3_turbo_file(artifact_path)
        paired: dict[str, dict[str, torch.Tensor]] = {}
        with safe_open(lora_file, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
            if metadata.get("key_format") != "minimax-h3-diffusers":
                raise ValueError(
                    f"Unsupported MiniMax-H3 Turbo key format {metadata.get('key_format')!r}; "
                    "expected 'minimax-h3-diffusers'"
                )
            raw_alpha = metadata.get("alpha")
            try:
                alpha = float(raw_alpha) if raw_alpha is not None else math.nan
            except ValueError as exc:
                raise ValueError(f"MiniMax-H3 Turbo alpha must be numeric, got {raw_alpha!r}") from exc
            if alpha != _H3_TURBO_ALPHA:
                raise ValueError(f"MiniMax-H3 Turbo v1.0 requires alpha={_H3_TURBO_ALPHA:g}, got {raw_alpha!r}")

            for key in checkpoint.keys():
                if key.endswith(_LORA_A_SUFFIX):
                    raw_target = key[: -len(_LORA_A_SUFFIX)]
                    side = "a"
                elif key.endswith(_LORA_B_SUFFIX):
                    raw_target = key[: -len(_LORA_B_SUFFIX)]
                    side = "b"
                else:
                    raise ValueError(f"Unconsumed MiniMax-H3 Turbo tensor: {key!r}")
                target = _normalize_h3_target(raw_target)
                target_pair = paired.setdefault(target, {})
                if side in target_pair:
                    raise ValueError(f"Duplicate MiniMax-H3 Turbo tensor for {target}.{side}")
                tensor = checkpoint.get_tensor(key)
                if side == "b" and target.endswith(".mlp.fc1"):
                    if tensor.shape[0] % 2:
                        raise ValueError(
                            f"MiniMax-H3 Turbo fc1 lora_B rows must split evenly, got {tuple(tensor.shape)}"
                        )
                    value, gate = tensor.chunk(2, dim=0)
                    tensor = torch.cat((gate, value), dim=0).contiguous()
                target_pair[side] = tensor

        updates: list[LowRankUpdate] = []
        for target, tensors in sorted(paired.items()):
            if set(tensors) != {"a", "b"}:
                raise ValueError(f"Incomplete MiniMax-H3 Turbo LoRA pair for {target}: {sorted(tensors)}")
            lora_a = tensors["a"]
            lora_b = tensors["b"]
            if lora_a.ndim != 2 or lora_b.ndim != 2 or lora_a.shape[0] != _H3_TURBO_RANK:
                raise ValueError(
                    f"MiniMax-H3 Turbo v1.0 requires rank {_H3_TURBO_RANK}, "
                    f"got A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)} for {target}"
                )
            if lora_b.shape[1] != _H3_TURBO_RANK:
                raise ValueError(f"MiniMax-H3 Turbo B rank mismatch for {target}: {tuple(lora_b.shape)}")
            updates.append(
                LowRankUpdate(
                    component="transformer",
                    logical_target=target,
                    lora_a=lora_a,
                    lora_b=lora_b,
                    intrinsic_scale=alpha / _H3_TURBO_RANK,
                )
            )
        if not updates:
            raise ValueError(f"MiniMax-H3 Turbo LoRA {lora_file} contains no supported updates")
        return LoadedDiffusionLoRA(name=deployment.name, updates=tuple(updates))


def create_minimax_h3_lora_loader(pipeline: nn.Module) -> MiniMaxH3TurboLoRALoader:
    return MiniMaxH3TurboLoRALoader(pipeline)


MINIMAX_H3_DIFFUSION_LORA_SUPPORT = DiffusionLoRASupport(
    loader_factory=create_minimax_h3_lora_loader,
    binding_plan=MINIMAX_H3_LORA_BINDING_PLAN,
    executor_factory=create_low_rank_executor,
    supports_composition=True,
)
