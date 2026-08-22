# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.models.minimax_h3.lora import (
    load_minimax_h3_turbo_lora,
)
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _request(path) -> LoRARequest:
    return LoRARequest(
        lora_name="turbo",
        lora_int_id=1,
        lora_path=str(path),
    )


def _write_tiny_turbo(path, *, alpha: str = "128", key_format: str = "minimax-h3-diffusers") -> None:
    rank = 128
    save_file(
        {
            "transformer_blocks.0.attn.to_q.lora_A.default.weight": torch.ones(rank, 2),
            "transformer_blocks.0.attn.to_q.lora_B.default.weight": torch.ones(3, rank),
            "transformer_blocks.0.ff.net.0.proj.lora_A.default.weight": torch.ones(rank, 2),
            "transformer_blocks.0.ff.net.0.proj.lora_B.default.weight": torch.cat(
                (torch.ones(2, rank), torch.full((2, rank), 2.0)),
                dim=0,
            ),
        },
        str(path),
        metadata={"alpha": alpha, "key_format": key_format},
    )


def test_h3_turbo_loads_through_legacy_lora_model_and_swaps_ffn(tmp_path):
    path = tmp_path / "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
    _write_tiny_turbo(path)

    loaded = load_minimax_h3_turbo_lora(
        partition="fl2va",
        lora_request=_request(path),
        lora_path=path,
        dtype=torch.float32,
    )

    assert loaded is not None
    lora_model, peft_helper = loaded
    assert peft_helper.r == 128
    assert peft_helper.lora_alpha == 128
    assert set(lora_model.loras) == {
        "blocks.0.attn.to_q",
        "blocks.0.mlp.fc1",
    }
    fc1 = lora_model.get_lora("blocks.0.mlp.fc1")
    assert fc1 is not None
    torch.testing.assert_close(fc1.lora_b[:2], torch.full((2, 128), 2.0))
    torch.testing.assert_close(fc1.lora_b[2:], torch.ones(2, 128))


def test_legacy_manager_uses_the_h3_model_loader_without_changing_its_interface(tmp_path):
    path = tmp_path / "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
    _write_tiny_turbo(path)

    class _Pipeline:
        def _load_diffusion_lora_adapter(self, **kwargs):
            return load_minimax_h3_turbo_lora(partition="fl2va", **kwargs)

    manager = object.__new__(DiffusionLoRAManager)
    manager.pipeline = _Pipeline()
    manager.dtype = torch.float32
    manager._expected_lora_modules = {"to_q", "fc1"}

    lora_model, peft_helper = manager._load_adapter(_request(path))

    assert lora_model.id == 1
    assert peft_helper.lora_alpha == 128
    assert set(lora_model.loras) == {"blocks.0.attn.to_q", "blocks.0.mlp.fc1"}


def test_h3_turbo_rejects_wrong_alpha_and_ref2va(tmp_path):
    wrong_alpha = tmp_path / "turbo_v1.0.safetensors"
    _write_tiny_turbo(wrong_alpha, alpha="8")
    with pytest.raises(ValueError, match="requires alpha=128"):
        load_minimax_h3_turbo_lora(
            partition="fl2va",
            lora_request=_request(wrong_alpha),
            lora_path=wrong_alpha,
            dtype=torch.float32,
        )

    valid = tmp_path / "turbo_valid_v1.0.safetensors"
    _write_tiny_turbo(valid)
    with pytest.raises(ValueError, match="supports FL2VA/T2VA only"):
        load_minimax_h3_turbo_lora(
            partition="ref2va",
            lora_request=_request(valid),
            lora_path=valid,
            dtype=torch.float32,
        )


def test_non_h3_checkpoint_falls_back_to_the_generic_peft_loader(tmp_path):
    path = tmp_path / "other.safetensors"
    _write_tiny_turbo(path, key_format="other")

    assert (
        load_minimax_h3_turbo_lora(
            partition="fl2va",
            lora_request=_request(path),
            lora_path=path,
            dtype=torch.float32,
        )
        is None
    )
