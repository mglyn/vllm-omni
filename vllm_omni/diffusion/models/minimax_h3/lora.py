# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import re

import torch
from vllm.model_executor.models.utils import WeightsMapper

from vllm_omni.diffusion.lora.plan import (
    DiffusionLoRAApplyPlan,
    DiffusionLoRALoadPlan,
)

_MINIMAX_H3_LORA_TARGETS = (
    "to_q",
    "to_k",
    "to_v",
    "qkv_proj",
    "out_proj",
    "fc1",
    "fc2",
)

MINIMAX_H3_LORA_APPLY_PLAN = DiffusionLoRAApplyPlan(
    component_names=("transformer",),
    target_modules=_MINIMAX_H3_LORA_TARGETS,
    packed_modules_mapping={"qkv_proj": ("to_q", "to_k", "to_v")},
)

_TURBO_LORA_CONFIG = {
    # The official MiniMax-H3-Turbo inference wrapper uses alpha=8 for its
    # rank-128 checkpoint. Keep this model-owned training-time scale separate
    # from the user-supplied composition coefficient.
    "lora_alpha": 8,
    "target_modules": list(_MINIMAX_H3_LORA_TARGETS),
}

_KREA_LORA_CONFIG = {
    # The Kohya checkpoint carries per-layer alpha tensors. The converter
    # folds alpha/r into B, so the shared loader must apply no second scale.
    "lora_alpha": None,
    "target_modules": list(_MINIMAX_H3_LORA_TARGETS),
}

_TURBO_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_substr={
        "token_refiner.refiner_blocks.": "token_refiner.blocks.",
        "transformer_blocks.": "blocks.",
        ".attn.to_out.0.": ".attn.out_proj.",
        ".ff.net.0.proj.": ".mlp.fc1.",
        ".ff.net.2.": ".mlp.fc2.",
        ".lora_A.default.": ".lora_A.",
        ".lora_B.default.": ".lora_B.",
    }
)

_KREA_QKV_KEY = re.compile(
    r"^lora_unet_blocks_(?P<block>\d+)_attn_qkv_proj\."
    r"(?P<kind>lora_down|lora_up)\.weight$"
)
_KREA_QKV_ALPHA_KEY = re.compile(r"^lora_unet_blocks_(?P<block>\d+)_attn_qkv_proj\.alpha$")


def _convert_krea_h3_lora(
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize the Kohya-style Krea H3 adapter and fold per-layer alpha."""

    converted: dict[str, torch.Tensor] = {}
    alphas: dict[str, float] = {}
    for name, tensor in tensors.items():
        if match := _KREA_QKV_KEY.fullmatch(name):
            module = f"blocks.{match.group('block')}.attn.qkv_proj"
            matrix = "lora_A" if match.group("kind") == "lora_down" else "lora_B"
            converted[f"{module}.{matrix}.weight"] = tensor
            continue
        if match := _KREA_QKV_ALPHA_KEY.fullmatch(name):
            module = f"blocks.{match.group('block')}.attn.qkv_proj"
            if tensor.numel() != 1:
                raise ValueError(f"MiniMax-H3 LoRA alpha must be scalar, got {name}={tuple(tensor.shape)}")
            alphas[module] = float(tensor.item())
            continue
        raise ValueError(f"Unsupported Krea MiniMax-H3 LoRA tensor: {name}")

    for module, alpha in alphas.items():
        a_name = f"{module}.lora_A.weight"
        b_name = f"{module}.lora_B.weight"
        if a_name not in converted or b_name not in converted:
            raise ValueError(f"MiniMax-H3 LoRA alpha has no complete A/B pair for {module}")
        rank = int(converted[a_name].shape[0])
        scaling = alpha / rank
        if scaling != 1.0:
            converted[b_name] = converted[b_name] * scaling
    return converted


def minimax_h3_lora_load_plan(
    adapter_path: str,
    tensor_keys: tuple[str, ...],
) -> DiffusionLoRALoadPlan | None:
    """Describe supported raw MiniMax-H3 LoRA checkpoint layouts."""

    del adapter_path
    if any(_KREA_QKV_KEY.fullmatch(key) for key in tensor_keys):
        return DiffusionLoRALoadPlan(
            peft_config=_KREA_LORA_CONFIG,
            state_dict_converter=_convert_krea_h3_lora,
        )
    if any(
        ("transformer_blocks." in key or "token_refiner.refiner_blocks." in key)
        and (".lora_A.default." in key or ".lora_B.default." in key)
        for key in tensor_keys
    ):
        return DiffusionLoRALoadPlan(
            peft_config=_TURBO_LORA_CONFIG,
            weights_mapper=_TURBO_WEIGHTS_MAPPER,
        )
    return None
