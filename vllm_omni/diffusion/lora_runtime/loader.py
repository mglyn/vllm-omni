# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from pathlib import Path

from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

from .types import DiffusionLoRADeployment


def resolve_diffusion_lora_artifact(deployment: DiffusionLoRADeployment) -> Path:
    """Resolve a deployment path once during worker startup."""

    local_path = Path(deployment.path).expanduser()
    if local_path.exists():
        return local_path.resolve()

    resolved = download_weights_from_hf_specific(
        deployment.path,
        cache_dir=None,
        allow_patterns=["*.safetensors"],
    )
    return Path(resolved)
