# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_minimax_h3_turbo_lora_plan_maps_lightx2v_keys() -> None:
    from vllm_omni.diffusion.models.minimax_h3.lora import (
        minimax_h3_lora_load_plan,
    )

    tensors = {
        "token_refiner.refiner_blocks.0.attn.to_q.lora_A.default.weight": torch.ones(2, 4),
        "token_refiner.refiner_blocks.0.attn.to_q.lora_B.default.weight": torch.ones(6, 2),
        "transformer_blocks.3.attn.to_out.0.lora_A.default.weight": torch.ones(2, 6),
        "transformer_blocks.3.attn.to_out.0.lora_B.default.weight": torch.ones(4, 2),
        "transformer_blocks.3.ff.net.0.proj.lora_A.default.weight": torch.ones(2, 4),
        "transformer_blocks.3.ff.net.0.proj.lora_B.default.weight": torch.ones(8, 2),
        "transformer_blocks.3.ff.net.2.lora_A.default.weight": torch.ones(2, 4),
        "transformer_blocks.3.ff.net.2.lora_B.default.weight": torch.ones(4, 2),
    }

    plan = minimax_h3_lora_load_plan("turbo.safetensors", tuple(tensors))

    assert plan is not None
    assert plan.state_dict_converter is None
    assert plan.peft_config["lora_alpha"] == 8
    assert plan.weights_mapper.apply_dict(tensors) == {
        "token_refiner.blocks.0.attn.to_q.lora_A.weight": tensors[
            "token_refiner.refiner_blocks.0.attn.to_q.lora_A.default.weight"
        ],
        "token_refiner.blocks.0.attn.to_q.lora_B.weight": tensors[
            "token_refiner.refiner_blocks.0.attn.to_q.lora_B.default.weight"
        ],
        "blocks.3.attn.out_proj.lora_A.weight": tensors["transformer_blocks.3.attn.to_out.0.lora_A.default.weight"],
        "blocks.3.attn.out_proj.lora_B.weight": tensors["transformer_blocks.3.attn.to_out.0.lora_B.default.weight"],
        "blocks.3.mlp.fc1.lora_A.weight": tensors["transformer_blocks.3.ff.net.0.proj.lora_A.default.weight"],
        "blocks.3.mlp.fc1.lora_B.weight": tensors["transformer_blocks.3.ff.net.0.proj.lora_B.default.weight"],
        "blocks.3.mlp.fc2.lora_A.weight": tensors["transformer_blocks.3.ff.net.2.lora_A.default.weight"],
        "blocks.3.mlp.fc2.lora_B.weight": tensors["transformer_blocks.3.ff.net.2.lora_B.default.weight"],
    }


def test_minimax_h3_lora_apply_plan_is_fl2va_only() -> None:
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline.partition = "combined"

    plan = pipeline.get_lora_apply_plan()

    assert plan.component_names == ("transformer",)
    assert "transformers_ref" not in plan.component_names
    assert plan.packed_modules_mapping == {"qkv_proj": ("to_q", "to_k", "to_v")}


def test_minimax_h3_ref2va_rejects_raw_fl2va_lora() -> None:
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline.partition = "ref2va"

    with pytest.raises(ValueError, match="Ref2VA-only"):
        pipeline.get_lora_load_plan(
            "turbo.safetensors",
            ("transformer_blocks.0.attn.to_q.lora_A.default.weight",),
        )


def test_minimax_h3_unknown_raw_lora_is_not_claimed() -> None:
    from vllm_omni.diffusion.models.minimax_h3.lora import (
        minimax_h3_lora_load_plan,
    )

    unsupported_layouts = (
        "unknown.lora_A.weight",
        "diffusion_model.blocks.3.attn.qkv_proj.lora_A.weight",
        "lora_unet_blocks_7_attn_qkv_proj.lora_down.weight",
    )
    for tensor_key in unsupported_layouts:
        assert minimax_h3_lora_load_plan("unknown.safetensors", (tensor_key,)) is None


def test_minimax_h3_turbo_four_nfe_uses_five_sigma_grid_points() -> None:
    from vllm_omni.diffusion.models.minimax_h3.time_request import (
        minimax_h3_time_shift_sigmas,
    )

    sigmas = minimax_h3_time_shift_sigmas(num_steps=5, shift_scale=12.0)

    assert len(sigmas) == 5
    assert len(sigmas) - 1 == 4
    assert sigmas[0] == 1.0
    assert sigmas[-1] == 0.0
