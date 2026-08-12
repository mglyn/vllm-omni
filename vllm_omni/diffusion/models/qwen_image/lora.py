# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
from diffusers.loaders.lora_conversion_utils import (
    _convert_non_diffusers_qwen_lora_to_diffusers,
)

from vllm_omni.diffusion.lora.plan import (
    DiffusionLoRAApplyPlan,
    DiffusionLoRALoadPlan,
)

_QWEN_IMAGE_LORA_TARGETS = (
    "to_q",
    "to_k",
    "to_v",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "to_out",
    "proj",
    "2",
)

QWEN_IMAGE_LORA_APPLY_PLAN = DiffusionLoRAApplyPlan(
    component_names=("transformer",),
    target_modules=_QWEN_IMAGE_LORA_TARGETS,
    packed_modules_mapping={
        "to_qkv": ("to_q", "to_k", "to_v"),
        "add_kv_proj": ("add_q_proj", "add_k_proj", "add_v_proj"),
    },
)


def _fold_diffusers_alpha(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fold per-layer Diffusers alpha values into the B matrices."""

    converted = dict(state_dict)
    for alpha_key in (key for key in state_dict if key.endswith(".alpha")):
        base_key = alpha_key.removesuffix(".alpha")
        lora_a_key = f"{base_key}.lora_A.weight"
        lora_b_key = f"{base_key}.lora_B.weight"
        if lora_a_key not in state_dict or lora_b_key not in state_dict:
            raise ValueError(f"LoRA alpha key {alpha_key!r} does not have matching A/B weights")
        rank = state_dict[lora_a_key].shape[0]
        converted[lora_b_key] = state_dict[lora_b_key] * (state_dict[alpha_key].item() / rank)
        converted.pop(alpha_key)
    return converted


def convert_qwen_image_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize published Qwen-Image LoRAs for the shared backend."""

    has_alpha = any(key.endswith(".alpha") for key in state_dict)
    diffusers_lora_keys = [key for key in state_dict if key.endswith((".lora_A.weight", ".lora_B.weight"))]
    component_prefixes = {key.startswith("transformer.") for key in diffusers_lora_keys}
    if len(component_prefixes) > 1:
        raise ValueError("Qwen-Image LoRA mixes component-prefixed and unprefixed Diffusers keys")
    is_component_prefixed = component_prefixes == {True}
    if has_alpha and diffusers_lora_keys:
        state_dict = _fold_diffusers_alpha(state_dict)

    if (has_alpha and not is_component_prefixed) or any(
        key.startswith(("diffusion_model.", "lora_unet_")) or ".lora_down.weight" in key for key in state_dict
    ):
        state_dict = _convert_non_diffusers_qwen_lora_to_diffusers(dict(state_dict))

    return {key.replace(".to_out.0.", ".to_out."): value for key, value in state_dict.items()}


def qwen_image_lora_load_plan(
    adapter_path: str,
    tensor_keys: tuple[str, ...],
) -> DiffusionLoRALoadPlan | None:
    del adapter_path
    if not any(key.endswith((".lora_A.weight", ".lora_down.weight")) for key in tensor_keys):
        return None
    return DiffusionLoRALoadPlan(
        peft_config={
            "lora_alpha": None,
            "target_modules": list(_QWEN_IMAGE_LORA_TARGETS),
        },
        state_dict_converter=convert_qwen_image_lora_state_dict,
    )


class QwenImageLoRAPlanMixin:
    def get_lora_apply_plan(self) -> DiffusionLoRAApplyPlan:
        return QWEN_IMAGE_LORA_APPLY_PLAN

    def get_lora_load_plan(
        self,
        adapter_path: str,
        tensor_keys: tuple[str, ...],
    ) -> DiffusionLoRALoadPlan | None:
        return qwen_image_lora_load_plan(adapter_path, tensor_keys)
