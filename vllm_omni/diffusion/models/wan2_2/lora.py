# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from pathlib import Path

import torch
from diffusers.loaders.lora_conversion_utils import (
    _convert_non_diffusers_wan_lora_to_diffusers,
)

from vllm_omni.diffusion.lora.plan import (
    DiffusionLoRAApplyPlan,
    DiffusionLoRALoadPlan,
)

_WAN_LORA_TARGETS = (
    "to_q",
    "to_k",
    "to_v",
    "to_out",
    "add_k_proj",
    "add_v_proj",
    "proj",
    "net_2",
)

WAN_LORA_APPLY_PLAN = DiffusionLoRAApplyPlan(
    component_names=("transformer", "transformer_2"),
    target_modules=_WAN_LORA_TARGETS,
    packed_modules_mapping={"to_qkv": ("to_q", "to_k", "to_v")},
)


def _wan_lora_component(adapter_path: str, has_transformer_2: bool) -> str:
    if not has_transformer_2:
        return "transformer"

    filename = Path(adapter_path).name.lower()
    if "high_noise" in filename:
        return "transformer"
    if "low_noise" in filename:
        return "transformer_2"
    raise ValueError(
        "Wan2.2 uses separate high- and low-noise transformers. The LoRA "
        "filename must contain 'high_noise' or 'low_noise' so it can be "
        "assigned without relying on adapter argument order."
    )


def convert_wan_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    component_name: str,
) -> dict[str, torch.Tensor]:
    """Normalize one published Wan LoRA for the selected transformer."""

    unsupported = [key for key in state_dict if key.endswith((".diff", ".diff_b", ".lora_B.bias"))]
    if unsupported:
        raise ValueError(
            "This Wan adapter contains dense or bias deltas, which are not "
            "representable by the shared low-rank LoRA backend: "
            f"{unsupported[:3]}"
        )

    if any(key.startswith("diffusion_model.") for key in state_dict):
        state_dict = _convert_non_diffusers_wan_lora_to_diffusers(dict(state_dict))

    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        key = key.replace(".ffn.net.0.", ".ffn.net_0.")
        key = key.replace(".ffn.net.2.", ".ffn.net_2.")
        key = key.replace(".to_out.0.", ".to_out.")
        if key.startswith("transformer."):
            key = f"{component_name}.{key.removeprefix('transformer.')}"
        elif not key.startswith(f"{component_name}."):
            key = f"{component_name}.{key}"
        converted[key] = value
    return converted


def wan_lora_load_plan(
    adapter_path: str,
    tensor_keys: tuple[str, ...],
    *,
    has_transformer_2: bool,
) -> DiffusionLoRALoadPlan | None:
    if not any(key.endswith((".lora_A.weight", ".lora_down.weight")) for key in tensor_keys):
        return None

    component_name = _wan_lora_component(adapter_path, has_transformer_2)

    def convert(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return convert_wan_lora_state_dict(state_dict, component_name=component_name)

    return DiffusionLoRALoadPlan(
        peft_config={
            "lora_alpha": None,
            "target_modules": list(_WAN_LORA_TARGETS),
        },
        state_dict_converter=convert,
    )


class WanLoRAPlanMixin:
    def get_lora_apply_plan(self) -> DiffusionLoRAApplyPlan:
        return WAN_LORA_APPLY_PLAN

    def get_lora_load_plan(
        self,
        adapter_path: str,
        tensor_keys: tuple[str, ...],
    ) -> DiffusionLoRALoadPlan | None:
        return wan_lora_load_plan(
            adapter_path,
            tensor_keys,
            has_transformer_2=bool(getattr(self, "has_transformer_2", False)),
        )
