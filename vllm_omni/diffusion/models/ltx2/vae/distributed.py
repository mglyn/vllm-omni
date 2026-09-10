# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright 2026 Lightricks and The HuggingFace Team. All rights reserved.
#
# Distributed tiling is copied and modified from Diffusers' serial LTX-2.5
# tiling at commit d035dcd7cc7c88e0a154609b62887d50bba9fdc2 (Apache-2.0).

"""LTX-2.5-specific distributed execution for the diffusion VAE decoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Any

import torch
import torch.distributed as dist
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedVaeMixin,
    GridSpec,
    TileTask,
)

from .decoder import LTX2VideoDiffusionDecoderModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _tile_intervals(length: int, tile_size: int, stride: int, min_size: int) -> list[tuple[int, int]]:
    """Build overlapping intervals, merging a too-small trailing remnant."""
    if length <= tile_size:
        return [(0, length)]
    starts = list(range(0, length, stride))
    while len(starts) > 1 and length - starts[-1] < min_size:
        starts.pop()
    return [(start, min(start + tile_size, length)) for start in starts[:-1]] + [(starts[-1], length)]


@dataclass(frozen=True)
class LTX2VideoDiffusionTilePlan:
    """Geometry shared by all ranks for one distributed DiffVAE decode."""

    temporal_tiles: tuple[tuple[int, int], ...]
    height_tiles: tuple[tuple[int, int], ...]
    width_tiles: tuple[tuple[int, int], ...]
    num_frames: int
    height: int
    width: int
    scale_t: int
    scale_h: int
    scale_w: int
    stride_t: int
    stride_h: int
    stride_w: int
    blend_frames: int
    blend_height: int
    blend_width: int
    single_step_x0: bool


@dataclass
class LTX2VideoDiffusionTileTask(TileTask):
    """One stage-4 + stage-5 tile and its rank-independent noise."""

    drop_leading_frame: bool = False
    crop_trailing_ghost: bool = False
    noise_generator: torch.Generator | list[torch.Generator] | None = None
    x_t: torch.Tensor | None = None
    output_shape: tuple[int, int, int, int, int] | None = None


class _LTX2VideoDiffusionStreamingMerger:
    """Reference-order spatiotemporal tile blending with bounded storage."""

    def __init__(
        self,
        model: DistributedLTX2VideoDiffusionDecoderModel,
        plan: LTX2VideoDiffusionTilePlan,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.plan = plan
        self.expected_coords = iter(
            product(
                range(len(plan.temporal_tiles)),
                range(len(plan.height_tiles)),
                range(len(plan.width_tiles)),
            )
        )
        self.num_tiles = len(plan.temporal_tiles) * len(plan.height_tiles) * len(plan.width_tiles)
        self.num_consumed = 0
        self.output = torch.empty(
            (
                batch_size,
                model.decoder.out_channels,
                plan.num_frames * plan.scale_t - (1 if plan.scale_t == 2 else 0),
                plan.height * plan.scale_h,
                plan.width * plan.scale_w,
            ),
            device=device,
            dtype=dtype,
        )
        self.output_frame_offset = 0
        self.current_t = -1
        self.current_h = -1
        self.previous_group: torch.Tensor | None = None
        self.current_group: torch.Tensor | None = None
        self.previous_row: list[torch.Tensor | None] = []
        self.current_row: list[torch.Tensor | None] = []

    def _finish_current_group(self) -> None:
        if self.current_group is None:
            return

        group = self.current_group
        if self.current_t > 0:
            if self.previous_group is None:
                raise RuntimeError(f"Missing temporal tile before LTX DiffVAE tile group {self.current_t}.")
            group = self.model.blend_t(self.previous_group, group, self.plan.blend_frames)

        if self.current_t < len(self.plan.temporal_tiles) - 1:
            keep_frames = self.plan.stride_t * self.plan.scale_t
            if self.current_t == 0 and self.plan.scale_t == 2:
                keep_frames -= 1
        else:
            keep_frames = group.shape[2]
        keep_frames = min(keep_frames, self.output.shape[2] - self.output_frame_offset)
        self.output[:, :, self.output_frame_offset : self.output_frame_offset + keep_frames].copy_(
            group[:, :, :keep_frames]
        )
        self.output_frame_offset += keep_frames
        self.previous_group = group
        self.current_group = None

    def _start_group(self, t_idx: int, tile: torch.Tensor) -> None:
        self._finish_current_group()
        self.current_t = t_idx
        self.current_h = -1
        self.previous_row = []
        self.current_row = []
        self.current_group = torch.empty(
            (
                tile.shape[0],
                tile.shape[1],
                tile.shape[2],
                self.plan.height * self.plan.scale_h,
                self.plan.width * self.plan.scale_w,
            ),
            device=tile.device,
            dtype=tile.dtype,
        )

    def add(self, coord: tuple[int, int, int], tile: torch.Tensor) -> None:
        expected = next(self.expected_coords)
        if coord != expected:
            raise ValueError(f"Expected LTX DiffVAE tile {expected}, got {coord}.")

        t_idx, h_idx, w_idx = coord
        if t_idx != self.current_t:
            self._start_group(t_idx, tile)
        if h_idx != self.current_h:
            if h_idx != self.current_h + 1 or w_idx != 0:
                raise ValueError(f"Unexpected LTX DiffVAE tile row transition at {coord}.")
            self.previous_row = self.current_row
            self.current_row = []
            self.current_h = h_idx

        if h_idx > 0:
            above = self.previous_row[w_idx]
            if above is None:
                raise RuntimeError(f"Missing tile above LTX DiffVAE tile {coord}.")
            overlap = self.plan.height_tiles[h_idx - 1][1] - self.plan.height_tiles[h_idx][0]
            tile = self.model.blend_v(above, tile, overlap * self.plan.scale_h)
            self.previous_row[w_idx] = None
        if w_idx > 0:
            left = self.current_row[w_idx - 1]
            if left is None:
                raise RuntimeError(f"Missing tile left of LTX DiffVAE tile {coord}.")
            overlap = self.plan.width_tiles[w_idx - 1][1] - self.plan.width_tiles[w_idx][0]
            tile = self.model.blend_h(left, tile, overlap * self.plan.scale_w)
        self.current_row.append(tile)

        keep_height = (
            (self.plan.height_tiles[h_idx + 1][0] - self.plan.height_tiles[h_idx][0]) * self.plan.scale_h
            if h_idx < len(self.plan.height_tiles) - 1
            else tile.shape[3]
        )
        keep_width = (
            (self.plan.width_tiles[w_idx + 1][0] - self.plan.width_tiles[w_idx][0]) * self.plan.scale_w
            if w_idx < len(self.plan.width_tiles) - 1
            else tile.shape[4]
        )
        core = tile[:, :, :, :keep_height, :keep_width]
        h_start = self.plan.height_tiles[h_idx][0] * self.plan.scale_h
        w_start = self.plan.width_tiles[w_idx][0] * self.plan.scale_w
        if self.current_group is None:
            raise RuntimeError(f"Missing output group for LTX DiffVAE tile {coord}.")
        self.current_group[
            :,
            :,
            :,
            h_start : h_start + keep_height,
            w_start : w_start + keep_width,
        ].copy_(core)

        self.num_consumed += 1

    def finish(self) -> torch.Tensor:
        if self.num_consumed != self.num_tiles:
            raise RuntimeError(f"Merged {self.num_consumed} LTX DiffVAE tiles, expected {self.num_tiles}.")
        self._finish_current_group()
        if self.output_frame_offset != self.output.shape[2]:
            raise RuntimeError(
                f"Merged {self.output_frame_offset} LTX DiffVAE frames, expected {self.output.shape[2]}."
            )
        self.previous_group = None
        self.previous_row = []
        self.current_row = []
        return self.output


class DistributedLTX2VideoDiffusionDecoderModel(LTX2VideoDiffusionDecoderModel, DistributedVaeMixin):
    """LTX-2.5 diffusion decoder with model-specific tile parallelism.

    Stages 1-3 run over the complete low-resolution feature volume on every
    participating rank. Stage 4 and the diffusion stage run as independent
    overlapping tile tasks, and rank 0 performs the reference blend/merge.
    """

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        model = super().from_pretrained(*args, **kwargs)
        model.init_distributed()
        return model

    def set_parallel_size(self, parallel_size: int, mode: str = "tile") -> None:
        if mode != "tile":
            raise ValueError(f"LTX-2.5 DiffVAE only supports vae_parallel_mode='tile', got {mode!r}.")
        super().set_parallel_size(parallel_size, mode=mode)

    def _build_tiled_decode_plan(
        self,
        features: torch.Tensor,
        num_inference_steps: int,
    ) -> LTX2VideoDiffusionTilePlan:
        """Build the rank-independent tile schedule around Diffusers' decoder."""
        decoder = self.decoder
        patch_size = decoder.patch_size
        upsample_stride = decoder.upsamples[-1].stride
        scale_t, scale_h, scale_w = (
            upsample_stride[0],
            upsample_stride[1] * patch_size,
            upsample_stride[2] * patch_size,
        )
        tile_t = self.tile_sample_min_num_frames // scale_t
        tile_h = self.tile_sample_min_height // scale_h
        tile_w = self.tile_sample_min_width // scale_w
        stride_t = self.tile_sample_stride_num_frames // scale_t
        stride_h = self.tile_sample_stride_height // scale_h
        stride_w = self.tile_sample_stride_width // scale_w
        min_sizes = [
            max(kernel_4, -(-kernel_5 // stride))
            for kernel_4, kernel_5, stride in zip(
                self.config.decoder_stage_kernels[-1],
                self.config.decoder_stage5_kernel,
                upsample_stride,
            )
        ]

        ghost_frames = decoder.trailing_pad_latent_frames * math.prod(
            upsample.stride[0] for upsample in decoder.upsamples[:-1]
        )
        num_frames = features.shape[1] - ghost_frames
        height, width = features.shape[2:4]
        return LTX2VideoDiffusionTilePlan(
            temporal_tiles=tuple(_tile_intervals(num_frames, tile_t, stride_t, min_sizes[0])),
            height_tiles=tuple(_tile_intervals(height, tile_h, stride_h, min_sizes[1])),
            width_tiles=tuple(_tile_intervals(width, tile_w, stride_w, min_sizes[2])),
            num_frames=num_frames,
            height=height,
            width=width,
            scale_t=scale_t,
            scale_h=scale_h,
            scale_w=scale_w,
            stride_t=stride_t,
            stride_h=stride_h,
            stride_w=stride_w,
            blend_frames=(tile_t - stride_t) * scale_t,
            blend_height=(tile_h - stride_h) * scale_h,
            blend_width=(tile_w - stride_w) * scale_w,
            single_step_x0=num_inference_steps == 1 and decoder.model_output_type == "x0",
        )

    @staticmethod
    def _iter_tiled_decode_coords(plan: LTX2VideoDiffusionTilePlan):
        for t_idx in range(len(plan.temporal_tiles)):
            for h_idx in range(len(plan.height_tiles)):
                for w_idx in range(len(plan.width_tiles)):
                    yield t_idx, h_idx, w_idx

    @staticmethod
    def _get_tiled_feature_slice(
        features: torch.Tensor,
        coord: tuple[int, int, int],
        plan: LTX2VideoDiffusionTilePlan,
    ) -> tuple[torch.Tensor, bool, bool]:
        t_idx, h_idx, w_idx = coord
        t0, t1 = plan.temporal_tiles[t_idx]
        h0, h1 = plan.height_tiles[h_idx]
        w0, w1 = plan.width_tiles[w_idx]
        is_origin = t0 == 0
        is_trailing = t1 == plan.num_frames
        feature_t1 = features.shape[1] if is_trailing else t1
        return features[:, t0:feature_t1, h0:h1, w0:w1], is_origin, is_trailing

    def _tiled_pixel_shape_from_features(
        self,
        feature_tile: torch.Tensor,
        *,
        drop_leading_frame: bool,
        crop_trailing_ghost: bool,
    ) -> tuple[int, int, int, int, int]:
        """Predict one output tile's shape without materializing stage 4."""
        decoder = self.decoder
        stride_t, stride_h, stride_w = decoder.upsamples[-1].stride
        num_frames = feature_tile.shape[1] * stride_t
        if stride_t == 2 and drop_leading_frame:
            num_frames -= 1
        if crop_trailing_ghost and decoder.trailing_pad_latent_frames > 0:
            content_frames = max(
                num_frames - decoder.trailing_pad_latent_frames * decoder.temporal_compression_ratio,
                1,
            )
            num_frames = min(num_frames, max(content_frames, decoder.stage5_kernel[0]))
        return (
            feature_tile.shape[0],
            decoder.out_channels,
            num_frames,
            feature_tile.shape[2] * stride_h * decoder.patch_size,
            feature_tile.shape[3] * stride_w * decoder.patch_size,
        )

    @staticmethod
    def _slice_tiled_noise(
        x_t_full: torch.Tensor,
        coord: tuple[int, int, int],
        tile_pixel_shape: tuple[int, int, int, int, int],
        plan: LTX2VideoDiffusionTilePlan,
    ) -> torch.Tensor:
        t_idx, h_idx, w_idx = coord
        t0, _ = plan.temporal_tiles[t_idx]
        h0, _ = plan.height_tiles[h_idx]
        w0, _ = plan.width_tiles[w_idx]
        pixel_t0 = t0 * plan.scale_t - (1 if t0 != 0 and plan.scale_t == 2 else 0)
        return x_t_full[
            :,
            :,
            pixel_t0 : pixel_t0 + tile_pixel_shape[2],
            h0 * plan.scale_h : h0 * plan.scale_h + tile_pixel_shape[3],
            w0 * plan.scale_w : w0 * plan.scale_w + tile_pixel_shape[4],
        ]

    def _merge_tiled_decode(
        self,
        tiles: dict[tuple[int, int, int], torch.Tensor],
        plan: LTX2VideoDiffusionTilePlan,
    ) -> torch.Tensor:
        """Blend distributed RGB tiles in Diffusers' serial order."""
        frame_groups = []
        for t_idx in range(len(plan.temporal_tiles)):
            rows = [
                [tiles[(t_idx, h_idx, w_idx)] for w_idx in range(len(plan.width_tiles))]
                for h_idx in range(len(plan.height_tiles))
            ]
            result_rows = []
            for h_idx, row in enumerate(rows):
                result_row = []
                for w_idx, tile in enumerate(row):
                    if h_idx > 0:
                        tile = self.blend_v(rows[h_idx - 1][w_idx], tile, plan.blend_height)
                    if w_idx > 0:
                        tile = self.blend_h(row[w_idx - 1], tile, plan.blend_width)
                    keep_height = plan.stride_h * plan.scale_h if h_idx < len(rows) - 1 else tile.shape[3]
                    keep_width = plan.stride_w * plan.scale_w if w_idx < len(row) - 1 else tile.shape[4]
                    result_row.append(tile[:, :, :, :keep_height, :keep_width])
                result_rows.append(torch.cat(result_row, dim=4))
            frame_groups.append(torch.cat(result_rows, dim=3))

        result = []
        for t_idx, group in enumerate(frame_groups):
            if t_idx > 0:
                group = self.blend_t(frame_groups[t_idx - 1], group, plan.blend_frames)
            if t_idx < len(frame_groups) - 1:
                keep_frames = plan.stride_t * plan.scale_t - (1 if t_idx == 0 and plan.scale_t == 2 else 0)
                group = group[:, :, :keep_frames]
            result.append(group)
        return torch.cat(result, dim=2)

    def _default_generator(self, device: torch.device) -> torch.Generator:
        if device.type == "cpu":
            return torch.default_generator
        device_module = getattr(torch, device.type, None)
        default_generators = getattr(device_module, "default_generators", None)
        if default_generators is None:
            raise ValueError(
                f"Distributed LTX-2.5 diffusion decode on {device.type!r} requires an explicit torch.Generator."
            )
        device_index = device.index
        if device_index is None:
            current_device = getattr(device_module, "current_device", None)
            device_index = current_device() if current_device is not None else 0
        return default_generators[device_index]

    def _sync_generators(
        self,
        generator: torch.Generator | list[torch.Generator] | None,
        device: torch.device,
    ) -> torch.Generator | list[torch.Generator]:
        """Make every rank start tile-noise generation from rank 0's state."""
        generators = self._default_generator(device) if generator is None else generator
        generator_list = generators if isinstance(generators, list) else [generators]
        for item in generator_list:
            state_on_device = item.get_state().to(device=device)
            dist.broadcast(state_on_device, src=0, group=self.distributed_executor.group)
            item.set_state(state_on_device.cpu())
        return generators

    @staticmethod
    def _clone_generator(generator: torch.Generator) -> torch.Generator:
        cloned = torch.Generator(device=generator.device)
        cloned.set_state(generator.get_state())
        return cloned

    def _clone_generators(
        self,
        generators: torch.Generator | list[torch.Generator],
    ) -> torch.Generator | list[torch.Generator]:
        if isinstance(generators, list):
            return [self._clone_generator(item) for item in generators]
        return self._clone_generator(generators)

    def _distributed_tile_split(
        self,
        z: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None,
        num_inference_steps: int,
    ) -> tuple[list[LTX2VideoDiffusionTileTask], GridSpec]:
        decoder = self.decoder
        features = decoder.forward_stages_1_to_3(z)
        plan = self._build_tiled_decode_plan(features, num_inference_steps)
        generators = self._sync_generators(generator, z.device)

        x_t_full = None
        if not plan.single_step_x0:
            pixel_frames = plan.num_frames * plan.scale_t - (1 if plan.scale_t == 2 else 0)
            x_t_full = randn_tensor(
                (
                    z.shape[0],
                    decoder.out_channels,
                    pixel_frames,
                    plan.height * plan.scale_h,
                    plan.width * plan.scale_w,
                ),
                generator=generators,
                device=z.device,
                dtype=z.dtype,
            )

        tasks = []
        for tile_id, coord in enumerate(self._iter_tiled_decode_coords(plan)):
            feature_tile, is_origin, is_trailing = self._get_tiled_feature_slice(features, coord, plan)
            tile_pixel_shape = self._tiled_pixel_shape_from_features(
                feature_tile,
                drop_leading_frame=is_origin,
                crop_trailing_ghost=is_trailing,
            )
            tile_generator = None
            x_t = None
            if plan.single_step_x0:
                # Save the canonical serial state for this tile, then advance
                # the request generator exactly as the single-rank tiled path
                # does. Task placement therefore cannot change tile noise.
                tile_generator = self._clone_generators(generators)
                discarded_noise = randn_tensor(
                    tile_pixel_shape,
                    generator=generators,
                    device=z.device,
                    dtype=z.dtype,
                )
                del discarded_noise
            else:
                x_t = self._slice_tiled_noise(x_t_full, coord, tile_pixel_shape, plan)

            tasks.append(
                LTX2VideoDiffusionTileTask(
                    tile_id=tile_id,
                    grid_coord=coord,
                    tensor=feature_tile,
                    workload=math.prod(tile_pixel_shape),
                    drop_leading_frame=is_origin,
                    crop_trailing_ghost=is_trailing,
                    noise_generator=tile_generator,
                    x_t=x_t,
                    output_shape=tile_pixel_shape,
                )
            )

        return tasks, GridSpec(
            split_dims=(2, 3, 4),
            grid_shape=(len(plan.temporal_tiles), len(plan.height_tiles), len(plan.width_tiles)),
            tile_spec={"plan": plan},
            output_dtype=z.dtype,
        )

    def _distributed_tile_exec(
        self,
        task: LTX2VideoDiffusionTileTask,
        num_inference_steps: int,
    ) -> torch.Tensor:
        context = self.decoder.forward_stage_4(
            task.tensor,
            drop_leading_frame=task.drop_leading_frame,
            crop_trailing_ghost=task.crop_trailing_ghost,
        )
        tile_pixel_shape = (
            context.shape[0],
            self.decoder.out_channels,
            context.shape[1],
            context.shape[2] * self.decoder.patch_size,
            context.shape[3] * self.decoder.patch_size,
        )
        if task.x_t is None:
            x_t = randn_tensor(
                tile_pixel_shape,
                generator=task.noise_generator,
                device=context.device,
                dtype=context.dtype,
            )
        else:
            x_t = task.x_t
        return self.decoder.denoise(context, x_t, num_inference_steps)

    def _distributed_tile_merge(
        self,
        tiles: dict[tuple[int, int, int], torch.Tensor],
        grid_spec: GridSpec,
    ) -> torch.Tensor:
        plan = grid_spec.tile_spec["plan"]
        if not isinstance(plan, LTX2VideoDiffusionTilePlan):
            raise TypeError(f"Expected an LTX2VideoDiffusionTilePlan, got {type(plan)!r}.")
        return self._merge_tiled_decode(tiles, plan)

    @staticmethod
    def _distributed_tile_output_shape(
        task: LTX2VideoDiffusionTileTask,
    ) -> tuple[int, int, int, int, int]:
        if task.output_shape is None:
            raise RuntimeError(f"Missing output shape for LTX DiffVAE tile {task.tile_id}.")
        return task.output_shape

    def _distributed_streaming_merger(
        self,
        grid_spec: GridSpec,
        z: torch.Tensor,
    ) -> _LTX2VideoDiffusionStreamingMerger:
        plan = grid_spec.tile_spec["plan"]
        if not isinstance(plan, LTX2VideoDiffusionTilePlan):
            raise TypeError(f"Expected an LTX2VideoDiffusionTilePlan, got {type(plan)!r}.")
        return _LTX2VideoDiffusionStreamingMerger(
            self,
            plan,
            batch_size=z.shape[0],
            device=z.device,
            dtype=grid_spec.output_dtype or z.dtype,
        )

    @staticmethod
    def _balance_distributed_tile_tasks(
        tasks: list[LTX2VideoDiffusionTileTask],
        num_ranks: int,
    ) -> list[list[LTX2VideoDiffusionTileTask]]:
        workloads: list[int | float] = [0] * num_ranks
        assigned: list[list[LTX2VideoDiffusionTileTask]] = [[] for _ in range(num_ranks)]
        for task in sorted(tasks, key=lambda item: item.workload, reverse=True):
            rank = workloads.index(min(workloads))
            assigned[rank].append(task)
            workloads[rank] += task.workload
        return assigned

    def _execute_distributed_streaming_decode(
        self,
        z: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None,
        num_inference_steps: int,
    ) -> torch.Tensor:
        """Decode local tiles and stream exact-size results to rank 0."""
        executor = self.distributed_executor
        parallel_size = min(executor.parallel_size, executor.world_size)
        tasks, grid_spec = self._distributed_tile_split(z, generator, num_inference_steps)
        assigned = self._balance_distributed_tile_tasks(tasks, parallel_size)
        local_tasks = assigned[executor.rank] if executor.rank < parallel_size else []
        local_results = {task.tile_id: self._distributed_tile_exec(task, num_inference_steps) for task in local_tasks}
        task_owner = {task.tile_id: owner for owner, rank_tasks in enumerate(assigned) for task in rank_tasks}
        output_dtype = grid_spec.output_dtype if grid_spec.output_dtype is not None else z.dtype
        merger = self._distributed_streaming_merger(grid_spec, z) if executor.rank == 0 else None

        for task in sorted(tasks, key=lambda item: item.tile_id):
            owner = task_owner[task.tile_id]
            if executor.rank == 0:
                if merger is None:
                    raise RuntimeError("Missing rank-0 LTX DiffVAE streaming merger.")
                if owner == 0:
                    tile = local_results.pop(task.tile_id)
                else:
                    tile = torch.empty(
                        self._distributed_tile_output_shape(task),
                        device=z.device,
                        dtype=output_dtype,
                    )
                    dist.recv(tile, src=owner, group=executor.group)
                merger.add(task.grid_coord, tile)
                del tile
            elif executor.rank == owner:
                tile = local_results.pop(task.tile_id)
                if not tile.is_contiguous():
                    tile = tile.contiguous()
                dist.send(tile, dst=0, group=executor.group)
                del tile

        if executor.rank == 0:
            if merger is None:
                raise RuntimeError("Missing rank-0 LTX DiffVAE streaming merger.")
            return merger.finish()
        return torch.empty(0, device=z.device, dtype=output_dtype)

    def tiled_decode(
        self,
        z: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        if not self.is_distributed_enabled():
            return super().tiled_decode(z, generator=generator, num_inference_steps=num_inference_steps)

        logger.debug("LTX-2.5 diffusion decoder running with distributed stage-4/stage-5 tiles")
        num_inference_steps = num_inference_steps or self.decoder.default_num_inference_steps
        result = self._execute_distributed_streaming_decode(
            z,
            generator,
            num_inference_steps,
        )
        if result.numel() == 0:
            # The base decode method crops five dimensions before the LTX
            # runtime discards non-output-rank results.
            return torch.empty((0, self.decoder.out_channels, 0, 0, 0), device=z.device, dtype=z.dtype)
        return result
