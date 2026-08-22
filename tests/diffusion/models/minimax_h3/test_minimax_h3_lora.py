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


def _write_tiny_turbo(
    path,
    *,
    alpha: str = "128",
    key_format: str = "minimax-h3-diffusers",
    omit_target: str | None = None,
) -> None:
    rank = 128
    tensors = {}
    target_suffixes = (
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    )
    for prefix, block_count in (
        ("transformer_blocks", 50),
        ("token_refiner.refiner_blocks", 2),
    ):
        for block_index in range(block_count):
            for suffix in target_suffixes:
                target = f"{prefix}.{block_index}.{suffix}"
                if target == omit_target:
                    continue
                tensors[f"{target}.lora_A.default.weight"] = torch.ones(rank, 2)
                if suffix == "ff.net.0.proj":
                    tensors[f"{target}.lora_B.default.weight"] = torch.cat(
                        (torch.ones(2, rank), torch.full((2, rank), 2.0)),
                        dim=0,
                    )
                else:
                    tensors[f"{target}.lora_B.default.weight"] = torch.ones(3, rank)
    save_file(
        tensors,
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
    assert len(lora_model.loras) == 312
    assert "blocks.0.attn.to_q" in lora_model.loras
    assert "blocks.0.mlp.fc1" in lora_model.loras
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
    assert len(lora_model.loras) == 312
    assert "blocks.0.attn.to_q" in lora_model.loras
    assert "blocks.0.mlp.fc1" in lora_model.loras


def test_h3_turbo_rejects_wrong_alpha_and_ref2va(tmp_path):
    wrong_alpha = tmp_path / "wrong_alpha" / "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
    wrong_alpha.parent.mkdir()
    _write_tiny_turbo(wrong_alpha, alpha="8")
    with pytest.raises(ValueError, match="requires alpha=128"):
        load_minimax_h3_turbo_lora(
            partition="fl2va",
            lora_request=_request(wrong_alpha),
            lora_path=wrong_alpha,
            dtype=torch.float32,
        )

    valid = tmp_path / "valid" / "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
    valid.parent.mkdir()
    _write_tiny_turbo(valid)
    with pytest.raises(ValueError, match="supports FL2VA/T2VA only"):
        load_minimax_h3_turbo_lora(
            partition="ref2va",
            lora_request=_request(valid),
            lora_path=valid,
            dtype=torch.float32,
        )


def test_h3_turbo_accepts_only_the_declared_v1_artifact(tmp_path):
    unsupported = tmp_path / "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"
    _write_tiny_turbo(unsupported)

    with pytest.raises(ValueError, match="supports only"):
        load_minimax_h3_turbo_lora(
            partition="fl2va",
            lora_request=_request(unsupported),
            lora_path=unsupported,
            dtype=torch.float32,
        )

    supported = tmp_path / "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
    _write_tiny_turbo(supported)
    # Directory resolution selects the declared artifact even when another
    # v1.0 checkpoint is present beside it.
    assert (
        load_minimax_h3_turbo_lora(
            partition="fl2va",
            lora_request=_request(tmp_path),
            lora_path=tmp_path,
            dtype=torch.float32,
        )
        is not None
    )


def test_h3_turbo_rejects_a_truncated_declared_artifact(tmp_path):
    path = tmp_path / "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
    _write_tiny_turbo(path, omit_target="transformer_blocks.49.ff.net.2")

    with pytest.raises(ValueError, match="target set does not match"):
        load_minimax_h3_turbo_lora(
            partition="fl2va",
            lora_request=_request(path),
            lora_path=path,
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
