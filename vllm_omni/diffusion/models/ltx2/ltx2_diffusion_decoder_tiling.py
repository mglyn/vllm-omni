# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Adaptive tile geometry for distributed LTX-2.5 DiffVAE decode."""

from __future__ import annotations

import math
from dataclasses import dataclass

AxisIntervals = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class LTX2AdaptiveTileGeometry:
    """A deterministic, equal-shape tile geometry for one distributed decode."""

    temporal_tiles: AxisIntervals
    height_tiles: AxisIntervals
    width_tiles: AxisIntervals
    reference_max_volume: int
    selected_max_volume: int
    workload_imbalance: float
    strategy: str

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return (len(self.temporal_tiles), len(self.height_tiles), len(self.width_tiles))

    @property
    def tile_count(self) -> int:
        return math.prod(self.grid_shape)


def legacy_tile_intervals(length: int, tile_size: int, stride: int, min_size: int) -> AxisIntervals:
    """Build the variable-edge intervals used by the native serial decoder."""
    if length <= tile_size:
        return ((0, length),)
    starts = list(range(0, length, stride))
    while len(starts) > 1 and length - starts[-1] < min_size:
        starts.pop()
    return tuple((start, min(start + tile_size, length)) for start in starts[:-1]) + ((starts[-1], length),)


def _equal_intervals(length: int, count: int, min_overlap: int, min_size: int) -> AxisIntervals | None:
    if count == 1:
        return ((0, length),) if length >= min_size else None

    tile_size = math.ceil((length + (count - 1) * min_overlap) / count)
    start_span = length - tile_size
    if tile_size < min_size or start_span < count - 1:
        return None

    short_gap, extra = divmod(start_span, count - 1)
    gaps = [short_gap + (index < extra) for index in range(count - 1)]
    if max(gaps) > tile_size - min_overlap:
        return None

    starts = [0]
    for gap in gaps:
        starts.append(starts[-1] + gap)
    return tuple((start, start + tile_size) for start in starts)


def _factorizations(product: int):
    for temporal_count in range(1, product + 1):
        if product % temporal_count:
            continue
        spatial_count = product // temporal_count
        for height_count in range(1, spatial_count + 1):
            if spatial_count % height_count == 0:
                yield temporal_count, height_count, spatial_count // height_count


def _tile_volumes(
    temporal_tiles: AxisIntervals,
    height_tiles: AxisIntervals,
    width_tiles: AxisIntervals,
    *,
    num_frames: int,
    ghost_frames: int,
) -> list[int]:
    volumes = []
    for temporal_start, temporal_end in temporal_tiles:
        temporal_size = temporal_end - temporal_start
        if temporal_end == num_frames:
            temporal_size += ghost_frames
        for height_start, height_end in height_tiles:
            for width_start, width_end in width_tiles:
                volumes.append(temporal_size * (height_end - height_start) * (width_end - width_start))
    return volumes


def _lpt_imbalance(workloads: list[int], parallel_size: int) -> float:
    rank_workloads = [0] * min(parallel_size, len(workloads))
    for workload in sorted(workloads, reverse=True):
        rank = rank_workloads.index(min(rank_workloads))
        rank_workloads[rank] += workload
    average = sum(rank_workloads) / len(rank_workloads)
    return max(rank_workloads) / average if average else 1.0


def _geometry_for_count(
    *,
    tile_count: int,
    num_frames: int,
    height: int,
    width: int,
    ghost_frames: int,
    parallel_size: int,
    min_overlap: tuple[int, int, int],
    min_sizes: tuple[int, int, int],
    reference_max_volume: int,
    max_aspect_ratio: float,
    max_workload_imbalance: float,
) -> LTX2AdaptiveTileGeometry | None:
    candidates: list[tuple[tuple[float, ...], LTX2AdaptiveTileGeometry]] = []
    for temporal_count, height_count, width_count in _factorizations(tile_count):
        temporal_tiles = _equal_intervals(num_frames, temporal_count, min_overlap[0], min_sizes[0])
        height_tiles = _equal_intervals(height, height_count, min_overlap[1], min_sizes[1])
        width_tiles = _equal_intervals(width, width_count, min_overlap[2], min_sizes[2])
        if temporal_tiles is None or height_tiles is None or width_tiles is None:
            continue

        tile_height = height_tiles[0][1] - height_tiles[0][0]
        tile_width = width_tiles[0][1] - width_tiles[0][0]
        aspect_ratio = max(tile_height / tile_width, tile_width / tile_height)
        if aspect_ratio > max_aspect_ratio:
            continue

        workloads = _tile_volumes(
            temporal_tiles,
            height_tiles,
            width_tiles,
            num_frames=num_frames,
            ghost_frames=ghost_frames,
        )
        selected_max_volume = max(workloads)
        imbalance = _lpt_imbalance(workloads, parallel_size)
        if selected_max_volume > reference_max_volume or imbalance > max_workload_imbalance:
            continue

        geometry = LTX2AdaptiveTileGeometry(
            temporal_tiles=temporal_tiles,
            height_tiles=height_tiles,
            width_tiles=width_tiles,
            reference_max_volume=reference_max_volume,
            selected_max_volume=selected_max_volume,
            workload_imbalance=imbalance,
            strategy="adaptive_equal",
        )
        duplicated_volume = sum(workloads)
        score = (
            float(temporal_count > 1),
            float(temporal_count),
            abs(math.log(tile_height / tile_width)),
            float(duplicated_volume),
        )
        candidates.append((score, geometry))

    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def plan_ltx2_distributed_tiles(
    *,
    num_frames: int,
    height: int,
    width: int,
    ghost_frames: int,
    parallel_size: int,
    default_tile: tuple[int, int, int],
    default_stride: tuple[int, int, int],
    min_sizes: tuple[int, int, int],
    max_tiles_per_rank: int = 4,
    max_aspect_ratio: float = 2.5,
    max_workload_imbalance: float = 1.1,
) -> LTX2AdaptiveTileGeometry:
    """Choose the fewest balanced tiles that do not raise per-tile memory.

    The native serial tile geometry defines the memory ceiling and overlap
    floor. Candidates start at one tile per participating rank, grow to at
    most four per rank, and avoid temporal splitting whenever a spatial grid
    is feasible. Small inputs may use fewer than ``parallel_size`` tiles.
    """
    if parallel_size < 1:
        raise ValueError(f"parallel_size must be positive, got {parallel_size}.")

    min_overlap = tuple(tile - stride for tile, stride in zip(default_tile, default_stride, strict=True))
    if any(overlap < 0 for overlap in min_overlap):
        raise ValueError(f"default_stride must not exceed default_tile: {default_stride} vs {default_tile}.")

    reference_tiles = (
        legacy_tile_intervals(num_frames, default_tile[0], default_stride[0], min_sizes[0]),
        legacy_tile_intervals(height, default_tile[1], default_stride[1], min_sizes[1]),
        legacy_tile_intervals(width, default_tile[2], default_stride[2], min_sizes[2]),
    )
    reference_volumes = _tile_volumes(
        *reference_tiles,
        num_frames=num_frames,
        ghost_frames=ghost_frames,
    )
    reference_max_volume = max(reference_volumes)

    for multiplier in range(1, max_tiles_per_rank + 1):
        geometry = _geometry_for_count(
            tile_count=parallel_size * multiplier,
            num_frames=num_frames,
            height=height,
            width=width,
            ghost_frames=ghost_frames,
            parallel_size=parallel_size,
            min_overlap=min_overlap,
            min_sizes=min_sizes,
            reference_max_volume=reference_max_volume,
            max_aspect_ratio=max_aspect_ratio,
            max_workload_imbalance=max_workload_imbalance,
        )
        if geometry is not None:
            return geometry

    for tile_count in range(parallel_size - 1, 0, -1):
        geometry = _geometry_for_count(
            tile_count=tile_count,
            num_frames=num_frames,
            height=height,
            width=width,
            ghost_frames=ghost_frames,
            parallel_size=parallel_size,
            min_overlap=min_overlap,
            min_sizes=min_sizes,
            reference_max_volume=reference_max_volume,
            max_aspect_ratio=max_aspect_ratio,
            max_workload_imbalance=max_workload_imbalance,
        )
        if geometry is not None:
            return geometry

    return LTX2AdaptiveTileGeometry(
        temporal_tiles=reference_tiles[0],
        height_tiles=reference_tiles[1],
        width_tiles=reference_tiles[2],
        reference_max_volume=reference_max_volume,
        selected_max_volume=reference_max_volume,
        workload_imbalance=_lpt_imbalance(reference_volumes, parallel_size),
        strategy="native_fallback",
    )
