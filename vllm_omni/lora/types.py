# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from vllm.lora.request import LoRARequest

from vllm_omni.lora.utils import stable_lora_int_id

LoRARequestInput: TypeAlias = LoRARequest | Sequence[LoRARequest] | None
LoRAScaleInput: TypeAlias = float | Sequence[float]


@dataclass(frozen=True)
class WeightedLoRA:
    """One adapter and its request/deployment-level mixing coefficient."""

    request: LoRARequest
    scale: float = 1.0

    @property
    def adapter_id(self) -> int:
        return self.request.lora_int_id


LoRAComposition: TypeAlias = tuple[WeightedLoRA, ...]
LoRACompositionKey: TypeAlias = tuple[tuple[int, float], ...]


def normalize_lora_composition(
    requests: LoRARequestInput,
    scales: LoRAScaleInput = 1.0,
) -> LoRAComposition:
    """Return a deterministic, validated composition.

    Duplicate adapter IDs are combined by adding their scales. Zero-scale
    entries are removed, and the result is sorted by adapter ID so every
    distributed rank binds the same concatenated low-rank layout.
    """

    if requests is None:
        return ()
    request_items = (requests,) if isinstance(requests, LoRARequest) else tuple(requests)
    if isinstance(scales, (int, float)):
        scale_items = (float(scales),) * len(request_items)
    else:
        scale_items = tuple(float(scale) for scale in scales)
        if len(scale_items) != len(request_items):
            raise ValueError(
                f"LoRA requests and scales must have the same length: {len(request_items)} != {len(scale_items)}"
            )

    combined: dict[int, WeightedLoRA] = {}
    for request, scale in zip(request_items, scale_items, strict=True):
        if not isinstance(request, LoRARequest):
            raise TypeError(f"Expected LoRARequest, got {type(request)!r}")
        if not math.isfinite(scale):
            raise ValueError(f"LoRA scale must be finite, got {scale!r}")

        previous = combined.get(request.lora_int_id)
        if previous is not None:
            if previous.request.lora_path != request.lora_path:
                raise ValueError(
                    f"LoRA adapter ID {request.lora_int_id} refers to both "
                    f"{previous.request.lora_path!r} and {request.lora_path!r}"
                )
            scale += previous.scale
        combined[request.lora_int_id] = WeightedLoRA(request=request, scale=scale)

    return tuple(adapter for _, adapter in sorted(combined.items()) if adapter.scale != 0.0)


def lora_composition_key(composition: LoRAComposition) -> LoRACompositionKey:
    return tuple((adapter.adapter_id, adapter.scale) for adapter in composition)


def split_lora_composition(
    composition: LoRAComposition,
) -> tuple[LoRARequest | tuple[LoRARequest, ...] | None, float | tuple[float, ...]]:
    """Project a canonical composition back to sampling-parameter fields."""

    if not composition:
        return None, 1.0
    if len(composition) == 1:
        return composition[0].request, composition[0].scale
    return tuple(adapter.request for adapter in composition), tuple(adapter.scale for adapter in composition)


def parse_lora_adapter_spec(value: str | Mapping[str, Any]) -> WeightedLoRA:
    """Parse ``PATH``, ``PATH=SCALE``, or a mapping startup specification."""

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("LoRA adapter specification must not be empty")
        if stripped.startswith("{"):
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                raise ValueError("LoRA JSON specification must be an object")
            value = parsed
        else:
            path = stripped
            scale = 1.0
            maybe_path, separator, maybe_scale = stripped.rpartition("=")
            if separator:
                try:
                    scale = float(maybe_scale)
                except ValueError:
                    pass
                else:
                    path = maybe_path
            value = {"path": path, "scale": scale}

    path_value = value.get("path") or value.get("lora_path") or value.get("local_path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("LoRA adapter specification requires a non-empty path")
    name_value = value.get("name") or value.get("lora_name") or Path(path_value).stem
    int_id_value = value.get("int_id") or value.get("lora_int_id") or stable_lora_int_id(path_value)
    scale_value = float(value.get("scale", value.get("lora_scale", 1.0)))
    composition = normalize_lora_composition(
        LoRARequest(
            lora_name=str(name_value),
            lora_int_id=int(int_id_value),
            lora_path=path_value,
        ),
        scale_value,
    )
    if not composition:
        raise ValueError("Startup LoRA adapter scale must be non-zero")
    return composition[0]


def parse_lora_adapter_specs(values: Sequence[str | Mapping[str, Any]] | None) -> LoRAComposition:
    if not values:
        return ()
    adapters = tuple(parse_lora_adapter_spec(value) for value in values)
    return normalize_lora_composition(
        tuple(adapter.request for adapter in adapters),
        tuple(adapter.scale for adapter in adapters),
    )
