# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from diffusers import AutoencoderKLLTX2Video
from diffusers.models.autoencoders.vae import DecoderOutput
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedOperator,
    DistributedVaeMixin,
    GridSpec,
    TileTask,
)

logger = init_logger(__name__)


class DistributedAutoencoderKLLTX2Video(AutoencoderKLLTX2Video, DistributedVaeMixin):
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        model = super().from_pretrained(*args, **kwargs)
        model.init_distributed()
        return model

    def tile_split(self, z: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        _, _, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        tiletask_list = []
        for i in range(0, height, tile_latent_stride_height):
            for j in range(0, width, tile_latent_stride_width):
                tile = z[:, :, :num_frames, i : i + tile_latent_min_height, j : j + tile_latent_min_width]
                tiletask_list.append(
                    TileTask(
                        len(tiletask_list),
                        (i // tile_latent_stride_height, j // tile_latent_stride_width),
                        tile,
                        workload=tile.shape[2] * tile.shape[3] * tile.shape[4],
                    )
                )

        tile_spec = {
            "sample_height": sample_height,
            "sample_width": sample_width,
            "blend_height": self.tile_sample_min_height - self.tile_sample_stride_height,
            "blend_width": self.tile_sample_min_width - self.tile_sample_stride_width,
            "tile_sample_stride_height": self.tile_sample_stride_height,
            "tile_sample_stride_width": self.tile_sample_stride_width,
        }
        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(tiletask_list[-1].grid_coord[0] + 1, tiletask_list[-1].grid_coord[1] + 1),
            tile_spec=tile_spec,
            output_dtype=self.dtype,
        )
        return tiletask_list, grid_spec

    def tile_exec(
        self,
        task: TileTask,
        temb: torch.Tensor | None = None,
        causal: bool | None = None,
    ) -> torch.Tensor:
        if hasattr(self, "clear_cache"):
            self.clear_cache()
        return self.decoder(task.tensor, temb, causal=causal)

    def tile_merge(self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec) -> torch.Tensor:
        grid_h, grid_w = grid_spec.grid_shape
        result_rows = []

        if hasattr(self, "clear_cache"):
            self.clear_cache()

        for i in range(grid_h):
            result_row = []
            for j in range(grid_w):
                tile = coord_tensor_map[(i, j)]
                if i > 0:
                    tile = self.blend_v(coord_tensor_map[(i - 1, j)], tile, grid_spec.tile_spec["blend_height"])
                if j > 0:
                    tile = self.blend_h(coord_tensor_map[(i, j - 1)], tile, grid_spec.tile_spec["blend_width"])
                result_row.append(
                    tile[
                        :,
                        :,
                        :,
                        : grid_spec.tile_spec["tile_sample_stride_height"],
                        : grid_spec.tile_spec["tile_sample_stride_width"],
                    ]
                )
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=3)[
            :, :, :, : grid_spec.tile_spec["sample_height"], : grid_spec.tile_spec["sample_width"]
        ]
        return dec

    def tiled_decode(
        self,
        z: torch.Tensor,
        temb: torch.Tensor | None = None,
        causal: bool | None = None,
        return_dict: bool = True,
    ):
        if not self.is_distributed_enabled():
            if causal is None:
                return super().tiled_decode(z, temb, return_dict=return_dict)
            return super().tiled_decode(z, temb, causal=causal, return_dict=return_dict)

        logger.debug("LTX2 video VAE decode running with distributed tiled executor")
        result = self.distributed_executor.execute(
            z,
            DistributedOperator(
                split=self.tile_split,
                exec=lambda task: self.tile_exec(task, temb=temb, causal=causal),
                merge=self.tile_merge,
            ),
            broadcast_result=True,
        )
        if not return_dict:
            return (result,)

        return DecoderOutput(sample=result)
