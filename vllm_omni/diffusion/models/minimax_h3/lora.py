# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

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
    # The published lightx2v inference wrapper uses alpha=8 for its
    # rank-128 checkpoint. Keep this model-owned training-time scale separate
    # from the user-supplied composition coefficient.
    "lora_alpha": 8,
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


def minimax_h3_lora_load_plan(
    adapter_path: str,
    tensor_keys: tuple[str, ...],
) -> DiffusionLoRALoadPlan | None:
    """Describe the supported lightx2v MiniMax-H3 Turbo checkpoint."""

    del adapter_path
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
