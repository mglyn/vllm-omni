# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

import pytest
from vllm.lora.request import LoRARequest

from vllm_omni.diffusion.lora.types import (
    lora_batch_key_fields,
    normalize_lora_composition,
    parse_lora_adapter_specs,
)
from vllm_omni.entrypoints.openai.utils import parse_lora_request

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request(adapter_id: int, path: str | None = None) -> LoRARequest:
    return LoRARequest(
        lora_name=f"adapter-{adapter_id}",
        lora_int_id=adapter_id,
        lora_path=path or f"/tmp/adapter-{adapter_id}",
    )


def test_composition_is_sorted_and_combines_duplicate_scales() -> None:
    composition = normalize_lora_composition(
        (_request(2), _request(1), _request(2)),
        (0.25, 0.5, 0.75),
    )

    assert [adapter.adapter_id for adapter in composition] == [1, 2]
    assert [adapter.scale for adapter in composition] == [0.5, 1.0]


def test_composition_rejects_invalid_scales_and_id_collisions() -> None:
    with pytest.raises(ValueError, match="same length"):
        normalize_lora_composition((_request(1), _request(2)), (1.0,))
    with pytest.raises(ValueError, match="finite"):
        normalize_lora_composition(_request(1), math.inf)
    with pytest.raises(ValueError, match="refers to both"):
        normalize_lora_composition(
            (_request(1, "/tmp/a"), _request(1, "/tmp/b")),
            (1.0, 1.0),
        )


def test_batch_key_fields_preserve_omitted_and_explicit_empty_semantics() -> None:
    assert lora_batch_key_fields(None) == (None, 1.0)
    assert lora_batch_key_fields((), ()) == ((), ())
    assert lora_batch_key_fields((_request(2), _request(1)), (0.25, 0.75)) == (
        (1, 2),
        (0.75, 0.25),
    )


def test_startup_specs_support_repeated_weighted_adapters() -> None:
    composition = parse_lora_adapter_specs(
        [
            "/tmp/adapter-a=0.25",
            '{"path":"/tmp/adapter-b","name":"style","scale":0.75,"int_id":22}',
        ]
    )

    by_name = {adapter.request.lora_name: adapter for adapter in composition}
    assert by_name["style"].scale == 0.75
    assert by_name["style"].adapter_id == 22
    assert by_name["adapter-a"].scale == 0.25


def test_request_parser_accepts_weighted_list_and_explicit_empty() -> None:
    requests, scales = parse_lora_request(
        [
            {"name": "turbo", "path": "/tmp/turbo", "scale": 0.8, "int_id": 8},
            {"name": "style", "path": "/tmp/style", "scale": 0.2, "int_id": 2},
        ]
    )

    assert isinstance(requests, tuple)
    assert [request.lora_int_id for request in requests] == [2, 8]
    assert scales == (0.2, 0.8)
    assert parse_lora_request([]) == ((), ())


def test_request_parser_preserves_scale_cancellation_as_explicit_empty() -> None:
    assert parse_lora_request(
        [
            {"name": "style", "path": "/tmp/style", "scale": 1.0, "int_id": 2},
            {"name": "style", "path": "/tmp/style", "scale": -1.0, "int_id": 2},
        ]
    ) == ((), ())
