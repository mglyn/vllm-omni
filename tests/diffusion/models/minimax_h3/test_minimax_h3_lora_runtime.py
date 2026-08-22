# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.lora_runtime import DiffusionLoRADeployment
from vllm_omni.diffusion.models.minimax_h3.lora_runtime import (
    MiniMaxH3TurboLoRALoader,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _write_tiny_h3_lora(path, *, alpha: str = "128") -> None:
    rank = 128
    save_file(
        {
            "transformer_blocks.0.ff.net.0.proj.lora_A.default.weight": torch.ones(rank, 2),
            "transformer_blocks.0.ff.net.0.proj.lora_B.default.weight": torch.cat(
                (torch.ones(2, rank), torch.full((2, rank), 2.0)), dim=0
            ),
            "transformer_blocks.0.attn.to_q.lora_A.default.weight": torch.ones(rank, 2),
            "transformer_blocks.0.attn.to_q.lora_B.default.weight": torch.ones(4, rank),
        },
        str(path),
        metadata={"alpha": alpha, "key_format": "minimax-h3-diffusers"},
    )


def test_h3_loader_normalizes_keys_alpha_and_ffn_layout(tmp_path):
    path = tmp_path / "minimax_h3_fl2v_turbo_4step_v1.0.safetensors"
    _write_tiny_h3_lora(path)
    loaded = MiniMaxH3TurboLoRALoader(SimpleNamespace(partition="fl2va")).load(
        DiffusionLoRADeployment("turbo", str(path)),
        path,
    )
    updates = {update.logical_target: update for update in loaded.updates}
    assert set(updates) == {"blocks.0.mlp.fc1", "blocks.0.attn.to_q"}
    assert updates["blocks.0.mlp.fc1"].rank == 128
    assert updates["blocks.0.mlp.fc1"].intrinsic_scale == 1.0
    torch.testing.assert_close(
        updates["blocks.0.mlp.fc1"].lora_b,
        torch.cat((torch.full((2, 128), 2.0), torch.ones(2, 128)), dim=0),
    )


def test_h3_loader_rejects_non_v1_alpha_and_ref2va(tmp_path):
    path = tmp_path / "turbo.safetensors"
    _write_tiny_h3_lora(path, alpha="8")
    loader = MiniMaxH3TurboLoRALoader(SimpleNamespace(partition="fl2va"))
    with pytest.raises(ValueError, match="requires alpha=128"):
        loader.load(DiffusionLoRADeployment("turbo", str(path)), path)

    with pytest.raises(ValueError, match="Ref2VA-only"):
        MiniMaxH3TurboLoRALoader(SimpleNamespace(partition="ref2va"))
