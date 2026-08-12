# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from vllm_omni.diffusion.lora.loader import DiffusionLoRAAdapterLoader
from vllm_omni.diffusion.lora.plan import (
    AdditiveBiasUpdate,
    ConvertedLoRAState,
    DiffusionAdapterUpdate,
    DiffusionLoRALoadPlan,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _CustomLoadPlanPipeline(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.custom_dit = torch.nn.Module()

    def get_lora_load_plan(
        self,
        adapter_path: str,
        tensor_keys: tuple[str, ...],
    ) -> DiffusionLoRALoadPlan | None:
        del adapter_path
        if "vendor.proj.down" not in tensor_keys:
            return None

        def convert(tensors: dict[str, torch.Tensor]) -> ConvertedLoRAState:
            updates = ()
            if (bias := tensors.get("vendor.proj.bias")) is not None:
                updates = (AdditiveBiasUpdate("proj", bias),)
            return ConvertedLoRAState(
                lora_tensors={
                    "proj.lora_A.weight": tensors["vendor.proj.down"],
                    "proj.lora_B.weight": tensors["vendor.proj.up"],
                },
                auxiliary_updates=updates,
            )

        return DiffusionLoRALoadPlan(
            peft_config={"lora_alpha": None, "target_modules": ["proj"]},
            state_dict_converter=convert,
        )


def test_model_owned_load_plan_describes_custom_checkpoint() -> None:
    loader = DiffusionLoRAAdapterLoader(
        pipeline=_CustomLoadPlanPipeline(),
        dtype=torch.bfloat16,
        expected_lora_modules={"proj"},
        component_names=("custom_dit",),
    )
    tensors = {
        "vendor.proj.down": torch.ones(2, 4),
        "vendor.proj.up": torch.ones(4, 2),
    }

    load_plan = loader.resolve_single_file_plan("custom.safetensors", tuple(tensors))

    assert load_plan.peft_config == {
        "lora_alpha": None,
        "target_modules": ["proj"],
    }
    assert load_plan.state_dict_converter is not None
    converted = load_plan.state_dict_converter(tensors)
    assert isinstance(converted, ConvertedLoRAState)
    assert set(converted.lora_tensors) == {"proj.lora_A.weight", "proj.lora_B.weight"}
    assert torch.equal(converted.lora_tensors["proj.lora_A.weight"], tensors["vendor.proj.down"])
    assert torch.equal(converted.lora_tensors["proj.lora_B.weight"], tensors["vendor.proj.up"])


def test_model_converter_returns_typed_auxiliary_updates() -> None:
    loader = DiffusionLoRAAdapterLoader(
        pipeline=_CustomLoadPlanPipeline(),
        dtype=torch.bfloat16,
        expected_lora_modules={"proj"},
        component_names=("custom_dit",),
    )
    bias = torch.ones(4)
    tensors = {
        "vendor.proj.down": torch.ones(2, 4),
        "vendor.proj.up": torch.ones(4, 2),
        "vendor.proj.bias": bias,
    }
    plan = loader.resolve_single_file_plan("custom.safetensors", tuple(tensors))
    assert plan.state_dict_converter is not None

    converted = plan.state_dict_converter(tensors)

    assert isinstance(converted, ConvertedLoRAState)
    assert len(converted.auxiliary_updates) == 1
    update = converted.auxiliary_updates[0]
    assert isinstance(update, AdditiveBiasUpdate)
    assert update.module_name == "proj"
    assert update.tensor is bias


@dataclass(frozen=True)
class _UnsupportedUpdate(DiffusionAdapterUpdate):
    tensor: torch.Tensor


def test_loader_rejects_unimplemented_auxiliary_update() -> None:
    update = _UnsupportedUpdate("proj", torch.ones(4))

    with pytest.raises(ValueError, match="unsupported auxiliary update _UnsupportedUpdate"):
        DiffusionLoRAAdapterLoader._validate_auxiliary_updates("custom.safetensors", (update,))
