# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LTX-2.5-specific distributed execution for the diffusion VAE decoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedOperator,
    DistributedVaeMixin,
    GridSpec,
    TileTask,
)

from .ltx2_diffusion_decoder import (
    LTX2VideoDiffusionDecoderModel,
    LTX2VideoDiffusionTilePlan,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class LTX2VideoDiffusionTileTask(TileTask):
    """One stage-4 + stage-5 tile and its rank-independent noise."""

    drop_leading_frame: bool = False
    crop_trailing_ghost: bool = False
    noise_generator: torch.Generator | list[torch.Generator] | None = None
    x_t: torch.Tensor | None = None


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
        result = self.distributed_executor.execute(
            z,
            DistributedOperator(
                split=lambda tensor: self._distributed_tile_split(tensor, generator, num_inference_steps),
                exec=lambda task: self._distributed_tile_exec(task, num_inference_steps),
                merge=self._distributed_tile_merge,
            ),
            broadcast_result=False,
        )
        if result.numel() == 0:
            # The base decode method crops five dimensions before the LTX
            # runtime discards non-output-rank results.
            return torch.empty((0, self.decoder.out_channels, 0, 0, 0), device=z.device, dtype=z.dtype)
        return result
