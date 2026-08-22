# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.diffusion.lora_runtime.types import (
    DiffusionLoRASelection,
    diffusion_lora_composition_key,
    normalize_diffusion_lora_composition,
    parse_diffusion_lora_deployments,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_parse_deployments_is_name_only_and_deterministic():
    deployments = parse_diffusion_lora_deployments(
        [
            '{"name":"style","path":"/models/style.safetensors"}',
            {"name": "turbo", "path": "org/turbo"},
        ]
    )
    assert [(item.name, item.path) for item in deployments] == [
        ("style", "/models/style.safetensors"),
        ("turbo", "org/turbo"),
    ]

    with pytest.raises(ValueError, match="unknown fields"):
        parse_diffusion_lora_deployments([{"name": "turbo", "path": "x", "int_id": 1}])
    with pytest.raises(ValueError, match="Duplicate"):
        parse_diffusion_lora_deployments(
            [
                {"name": "turbo", "path": "a"},
                {"name": "turbo", "path": "b"},
            ]
        )


def test_composition_combines_by_name_and_rejects_paths():
    composition = normalize_diffusion_lora_composition(
        [
            {"name": "turbo", "scale": 0.75},
            DiffusionLoRASelection("style", 0.5),
            {"name": "turbo", "scale": 0.25},
        ]
    )
    assert composition == (
        DiffusionLoRASelection("style", 0.5),
        DiffusionLoRASelection("turbo", 1.0),
    )
    assert diffusion_lora_composition_key(composition) == (("style", 0.5), ("turbo", 1.0))

    with pytest.raises(ValueError, match="unknown fields"):
        normalize_diffusion_lora_composition([{"name": "turbo", "path": "/tmp/turbo"}])
