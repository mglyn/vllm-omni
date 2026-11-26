# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Candidate implementations shared by offline tuning and eager dispatch."""

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .profile import Target


@dataclass(frozen=True)
class Tile:
    backend: ClassVar[str] = "triton"
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int

    def __post_init__(self):
        for value in (self.block_m, self.block_n, self.block_k):
            if type(value) is not int or value not in (16, 32, 64, 128, 256):
                raise ValueError(f"Invalid GEMM tile dimension: {value}")
        if type(self.num_warps) is not int or self.num_warps not in (4, 8):
            raise ValueError("num_warps must be 4 or 8")
        if type(self.num_stages) is not int or not 1 <= self.num_stages <= 5:
            raise ValueError("num_stages must be in [1, 5]")

    @classmethod
    def candidates(cls, target: "Target", shape: tuple[int, int, int]):
        return [
            cls(m, n, k, warps, stages)
            for m, n in ((16, 64), (32, 64), (64, 64), (64, 128), (128, 128))
            for k in (32, 64)
            for warps, stages in ((4, 3), (8, 4))
        ]

    def run(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        from .triton_mm import linear

        return linear(x, weight, self)


@dataclass(frozen=True)
class PadRows:
    backend: ClassVar[str] = "pad_rows"
    multiple: int

    def __post_init__(self):
        if type(self.multiple) is not int or self.multiple not in (16, 64, 128, 256):
            raise ValueError("Invalid GEMM row alignment")

    @classmethod
    def candidates(cls, target: "Target", shape: tuple[int, int, int]):
        rows, _, _ = shape
        alignments: dict[int, PadRows] = {}
        for multiple in (16, 64, 128, 256):
            padding = -rows % multiple
            if padding:
                alignments.setdefault(padding, cls(multiple))
        return list(alignments.values())

    def run(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        flat = x.view(-1, weight.size(1))
        rows = flat.size(0)
        padded = F.pad(flat, (0, 0, 0, -rows % self.multiple))
        return F.linear(padded, weight)[:rows].view(*x.shape[:-1], weight.size(0))


GemmConfig = Tile | PadRows
BACKENDS: dict[str, type[Tile] | type[PadRows]] = {cls.backend: cls for cls in (Tile, PadRows)}


def entry(config: GemmConfig) -> dict:
    return {"backend": config.backend, "config": asdict(config)}
