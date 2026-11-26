# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""On-disk contract shared by the offline tuner and inference workers."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import vllm
from vllm.triton_utils import triton

from vllm_omni.platforms import current_omni_platform

from .backends import BACKENDS, GemmConfig

OPERATION = "bf16_nt_no_bias"


@dataclass(frozen=True)
class Target:
    platform: str
    device: str
    arch: str
    torch_version: str
    triton_version: str
    runtime: str
    compute_units: int
    vllm_version: str

    @classmethod
    def current(cls, device: torch.device) -> "Target":
        index = device.index if device.index is not None else torch.accelerator.current_device_index()
        with torch.accelerator.device_index(index):
            target = triton.runtime.driver.active.get_current_target()
            compute_units = triton.runtime.driver.active.utils.get_device_properties(index)["multiprocessor_count"]
        return cls(
            target.backend,
            current_omni_platform.get_device_name(index),
            str(target.arch),
            str(torch.__version__),
            str(triton.__version__),
            str(torch.version.hip or torch.version.cuda),
            compute_units,
            str(vllm.__version__),
        )

    @property
    def filename(self) -> str:
        digest = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]
        name = "".join(c if c.isalnum() else "_" for c in self.device)
        return f"{name}-{self.arch}-bf16-nt-{digest}.json"


def shape_key(shape: tuple[int, int, int]) -> str:
    return ",".join(map(str, shape))


def parse_shape(key: str) -> tuple[int, int, int]:
    parts = key.split(",")
    if len(parts) != 3 or not all(p.isascii() and p.isdecimal() and int(p) > 0 for p in parts):
        raise ValueError(f"Expected a positive M,N,K triple, got {key!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def profile_entries(data: dict, target: Target) -> dict[tuple[int, int, int], GemmConfig]:
    if data.get("version") != 1 or data.get("target") != asdict(target):
        raise ValueError("Incompatible GEMM tuning target or schema")
    if data.get("operation") != OPERATION:
        raise ValueError("Incompatible GEMM operation")
    entries: dict[tuple[int, int, int], GemmConfig] = {}
    for key, row in data["entries"].items():
        shape = parse_shape(key)
        if shape_key(shape) != key:
            raise ValueError(f"Noncanonical GEMM shape {key!r}")
        name = row["backend"]
        if name == "default":
            continue
        if name not in BACKENDS:
            raise ValueError(f"Unknown GEMM backend {name!r}")
        entries[shape] = BACKENDS[name](**row["config"])
    return entries


def load_configs(directory: Path, target: Target) -> dict[tuple[int, int, int], GemmConfig]:
    path = directory / target.filename
    if not path.is_file():
        return {}
    return profile_entries(json.loads(path.read_text()), target)
