# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LTX-2.5 Diffusion Fidelity Rendering (DFR), spatial 4K path.

This entry implements the official ``spatial_upscalings=2`` flow only: a
quarter-resolution generated-keyframe pass, a half-resolution IC-LoRA detail
pass, and a 2x2 tiled full-resolution epilogue. Temporal upscaling is outside
this pipeline's request contract.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import numpy as np
import PIL.Image
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents

from . import ltx2_latents as latent_ops
from .ltx2_components import LTX25_DFR_COMPONENT_PROFILE
from .ltx2_conditioning import LTXI2VConditioningMixin, _preprocess_i2v_pil_images
from .ltx2_denoise import (
    LTXDenoiseContext,
    LTXForwardContext,
    LTXGeneratedKeyframeLayout,
    LTXPhaseResult,
    LTXPreparedVideoState,
)
from .ltx2_recipes import LTX25_DFR_RECIPE, LTXPhaseRecipe
from .ltx2_request import LTXRequestInputs
from .ltx2_runtime import LTXRuntime

_DFR_SEGMENT_CANDIDATES = (24, 32)
_DFR_SPATIAL_OVERLAP = 12


def resolve_dfr_canvas(num_frames: int, temporal_scale: int = 8) -> tuple[int, tuple[int, ...]]:
    """Return the padded DFR canvas and its generated keyframe positions."""
    if num_frames < 2 or (num_frames - 1) % temporal_scale:
        raise ValueError(f"DFR num_frames must be {temporal_scale} * k + 1 and at least 2, got {num_frames}.")
    content = num_frames - 1
    segment = min(
        _DFR_SEGMENT_CANDIDATES,
        key=lambda candidate: ((-content) % candidate, -candidate),
    )
    padded_content = content + (-content) % segment
    positions = tuple(range(segment, padded_content + 1, segment))
    return padded_content + 1, positions


@dataclass(frozen=True)
class _DFRVideoConditioning:
    generated_positions: tuple[int, ...] = ()
    generated_initials: torch.Tensor | None = None
    reference_latent: torch.Tensor | None = None
    reference_downscale_factor: int = 1
    encoded_keyframes: torch.Tensor | None = None
    encoded_keyframe_positions: tuple[int, ...] = ()
    tiled: bool = False


@dataclass(frozen=True)
class _TileInterval:
    start: int
    end: int
    left_ramp: int
    right_ramp: int


def _split_two_tiles(length: int, overlap: int) -> tuple[_TileInterval, ...]:
    if length < 2:
        return (_TileInterval(0, length, 0, 0),)
    overlap = min(overlap, length - 2)
    total = length + overlap
    tile_size, remainder = divmod(total, 2)
    first_size = tile_size + (1 if remainder else 0)
    return (
        _TileInterval(0, first_size, 0, overlap),
        _TileInterval(first_size - overlap, length, overlap, 0),
    )


def _trapezoid(length: int, left: int, right: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.ones(length, device=device, dtype=dtype)
    if left:
        mask[:left] = torch.linspace(0, 1, left + 2, device=device, dtype=dtype)[1:-1]
    if right:
        mask[-right:] = torch.linspace(1, 0, right + 2, device=device, dtype=dtype)[1:-1]
    return mask


def _lanczos_x2_frame(frame: torch.Tensor) -> torch.Tensor:
    """Resize one ``(C,H,W)`` VAE-range frame with official PIL Lanczos."""
    image_array = (
        ((frame.detach().float().clamp(-1, 1) + 1) * 127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    image = PIL.Image.fromarray(image_array, mode="RGB")
    image = image.resize((image.width * 2, image.height * 2), resample=PIL.Image.Resampling.LANCZOS)
    resized = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1)
    return resized / 127.5 - 1


class LTX25DFRPipeline(LTXI2VConditioningMixin, LTXRuntime):
    """Official LTX-2.5 T/I2VA DFR path for 3840x2176 output at 24 fps."""

    pipeline_kind = "dfr"
    component_profile = LTX25_DFR_COMPONENT_PROFILE
    pipeline_recipe = LTX25_DFR_RECIPE
    _dit_modules: ClassVar[list[str]] = list(LTX25_DFR_COMPONENT_PROFILE.dit_modules)
    _encoder_modules: ClassVar[list[str]] = list(LTX25_DFR_COMPONENT_PROFILE.encoder_modules)
    _vae_modules: ClassVar[list[str]] = list(LTX25_DFR_COMPONENT_PROFILE.vae_modules)
    _resident_modules: ClassVar[list[str]] = list(LTX25_DFR_COMPONENT_PROFILE.resident_modules)
    supports_request_batch = False
    support_image_input = True
    unified_text_image_entry = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.model_version != "2.5":
            raise ValueError("LTX25DFRPipeline requires an LTX-2.5 distilled checkpoint.")
        if not getattr(self.transformer.config, "use_keyframes_abs_pos_embedding", False):
            raise ValueError(
                "LTX25DFRPipeline requires a checkpoint whose transformer config enables "
                "`use_keyframes_abs_pos_embedding`."
            )
        self._dfr_conditioning: _DFRVideoConditioning | None = None

    @staticmethod
    def _resolve_request_phase_sigmas(
        req: Any,
        stage_1_fallback: list[float] | None,
        stage_2_fallback: list[float] | None,
    ) -> tuple[list[float] | None, list[float] | None, list[float] | None] | None:
        pair = LTXRuntime._resolve_request_phase_sigmas(req, stage_1_fallback, stage_2_fallback)
        return None if pair is None else (pair[0], pair[1], pair[1])

    def _prepare_image_pixels(
        self,
        image: Any,
        request_inputs: LTXRequestInputs,
        prompt_dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        image_crf = 18 if request_inputs.image_crf is None else int(request_inputs.image_crf)
        is_pil = isinstance(image, PIL.Image.Image) or (
            isinstance(image, list) and image and all(isinstance(item, PIL.Image.Image) for item in image)
        )
        if image_crf and not is_pil:
            raise ValueError("DFR `image_crf` requires PIL images; use image_crf=0 for tensor input.")
        if is_pil:
            return _preprocess_i2v_pil_images(
                image,
                height=request_inputs.height,
                width=request_inputs.width,
                crf=image_crf,
                device=device,
                dtype=prompt_dtype,
            )
        if isinstance(image, torch.Tensor):
            pixels = image.unsqueeze(0) if image.ndim == 3 else image
        elif isinstance(image, list) and image and isinstance(image[0], torch.Tensor):
            pixels = torch.stack(image)
        else:
            pixels = self.video_processor.preprocess(
                image,
                height=request_inputs.height,
                width=request_inputs.width,
            )
        return pixels.to(device=device, dtype=prompt_dtype)

    def _append_keyframe_tokens(
        self,
        *,
        normalized_keyframes: torch.Tensor,
        positions: tuple[int, ...],
        fps: float,
        latent_chunks: list[torch.Tensor],
        clean_chunks: list[torch.Tensor],
        denoise_chunks: list[torch.Tensor],
        coord_chunks: list[torch.Tensor],
        keyframe_mask_chunks: list[torch.Tensor],
    ) -> None:
        if normalized_keyframes.shape[2] != len(positions):
            raise ValueError(
                f"DFR has {normalized_keyframes.shape[2]} encoded keyframes for {len(positions)} positions."
            )
        batch, _, _, height, width = normalized_keyframes.shape
        for index, pixel_frame in enumerate(positions):
            frame = normalized_keyframes[:, :, index : index + 1]
            tokens = latent_ops.pack_latents(
                frame,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            )
            coords = self.transformer.rope.prepare_video_coords(batch, 1, height, width, frame.device, fps=fps)
            coords[:, 0, ...] = float(pixel_frame) / fps
            coords[:, 0, ..., 1] = float(pixel_frame + 1) / fps
            latent_chunks.append(torch.zeros_like(tokens))
            clean_chunks.append(tokens)
            denoise_chunks.append(tokens.new_zeros((*tokens.shape[:2], 1)))
            coord_chunks.append(coords)
            keyframe_mask_chunks.append(tokens.new_zeros((*tokens.shape[:2], 1)))

    def _prepare_video_latents_stage(
        self,
        request_inputs: LTXRequestInputs,
        prompt_context: Any,
        *,
        device: torch.device,
        noise_scale: float,
        image: Any | None = None,
    ) -> LTXPreparedVideoState:
        conditioning = self._dfr_conditioning
        if conditioning is None:
            raise RuntimeError("DFR phase conditioning was not installed before latent preparation.")

        batch = prompt_context.batch_size * request_inputs.num_videos_per_prompt
        latent_frames, latent_height, latent_width = latent_ops.resolve_video_latent_shape(
            request_inputs.height,
            request_inputs.width,
            request_inputs.num_frames,
            vae_spatial_compression_ratio=self.vae_spatial_compression_ratio,
            vae_temporal_compression_ratio=self.vae_temporal_compression_ratio,
        )
        latent_shape = (
            batch,
            self.transformer.config.in_channels,
            latent_frames,
            latent_height,
            latent_width,
        )
        if request_inputs.latents is None:
            base = torch.zeros(latent_shape, device=device, dtype=prompt_context.positive_connector_prompt_embeds.dtype)
        else:
            if tuple(request_inputs.latents.shape) != latent_shape:
                raise ValueError(
                    f"DFR phase latent has shape {tuple(request_inputs.latents.shape)}, expected {latent_shape}."
                )
            base = latent_ops.normalize_latents(
                request_inputs.latents.to(device=device),
                self.vae.latents_mean,
                self.vae.latents_std,
                self.vae.config.scaling_factor,
            ).to(prompt_context.positive_connector_prompt_embeds.dtype)

        base_tokens = latent_ops.pack_latents(
            base,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )
        base_token_count = base_tokens.shape[1]
        latent_chunks = [base_tokens]
        clean_chunks = [base_tokens.clone()]
        denoise_chunks = [base_tokens.new_ones((*base_tokens.shape[:2], 1))]
        coord_chunks = [
            self.transformer.rope.prepare_video_coords(
                batch,
                latent_frames,
                latent_height,
                latent_width,
                device,
                fps=request_inputs.frame_rate,
            )
        ]
        first_frame_tokens = base_token_count // latent_frames
        base_keyframes_mask = base_tokens.new_zeros((*base_tokens.shape[:2], 1))
        base_keyframes_mask[:, :first_frame_tokens] = 1
        keyframe_mask_chunks = [base_keyframes_mask]

        if image is not None:
            pixels = self._prepare_image_pixels(
                image,
                request_inputs,
                prompt_context.positive_connector_prompt_embeds.dtype,
                device,
            )
            encoded_image = self._encode_i2v_image_latents(
                pixels,
                batch_size=batch,
                generator=request_inputs.generator,
                dtype=prompt_context.positive_connector_prompt_embeds.dtype,
            )
            self._append_keyframe_tokens(
                normalized_keyframes=encoded_image,
                positions=(0,),
                fps=request_inputs.frame_rate,
                latent_chunks=latent_chunks,
                clean_chunks=clean_chunks,
                denoise_chunks=denoise_chunks,
                coord_chunks=coord_chunks,
                keyframe_mask_chunks=keyframe_mask_chunks,
            )

        if conditioning.encoded_keyframes is not None:
            self._append_keyframe_tokens(
                normalized_keyframes=conditioning.encoded_keyframes.to(device=device),
                positions=conditioning.encoded_keyframe_positions,
                fps=request_inputs.frame_rate,
                latent_chunks=latent_chunks,
                clean_chunks=clean_chunks,
                denoise_chunks=denoise_chunks,
                coord_chunks=coord_chunks,
                keyframe_mask_chunks=keyframe_mask_chunks,
            )

        generated_layout = None
        if conditioning.generated_positions:
            tokens_per_keyframe = first_frame_tokens
            first_token = sum(chunk.shape[1] for chunk in latent_chunks)
            if conditioning.generated_initials is None:
                slot_tokens = base_tokens.new_zeros(
                    batch,
                    len(conditioning.generated_positions) * tokens_per_keyframe,
                    base_tokens.shape[2],
                )
            else:
                initials = latent_ops.normalize_latents(
                    conditioning.generated_initials.to(device=device),
                    self.vae.latents_mean,
                    self.vae.latents_std,
                    self.vae.config.scaling_factor,
                ).to(base_tokens.dtype)
                expected = (batch, base.shape[1], len(conditioning.generated_positions), latent_height, latent_width)
                if tuple(initials.shape) != expected:
                    raise ValueError(f"DFR generated keyframes have shape {tuple(initials.shape)}, expected {expected}.")
                slot_tokens = torch.cat(
                    [
                        latent_ops.pack_latents(
                            initials[:, :, index : index + 1],
                            self.transformer_spatial_patch_size,
                            self.transformer_temporal_patch_size,
                        )
                        for index in range(initials.shape[2])
                    ],
                    dim=1,
                )
            slot_coords = []
            for pixel_frame in conditioning.generated_positions:
                coords = self.transformer.rope.prepare_video_coords(
                    batch,
                    1,
                    latent_height,
                    latent_width,
                    device,
                    fps=request_inputs.frame_rate,
                )
                coords[:, 0, ...] = float(pixel_frame) / request_inputs.frame_rate
                coords[:, 0, ..., 1] = float(pixel_frame + 1) / request_inputs.frame_rate
                slot_coords.append(coords)
            latent_chunks.append(slot_tokens)
            clean_chunks.append(torch.zeros_like(slot_tokens))
            denoise_chunks.append(slot_tokens.new_ones((*slot_tokens.shape[:2], 1)))
            coord_chunks.append(torch.cat(slot_coords, dim=2))
            keyframe_mask_chunks.append(slot_tokens.new_ones((*slot_tokens.shape[:2], 1)))
            generated_layout = LTXGeneratedKeyframeLayout(
                pixel_frame_indices=conditioning.generated_positions,
                tokens_per_keyframe=tokens_per_keyframe,
                first_token=first_token,
            )

        if conditioning.reference_latent is not None:
            reference = latent_ops.normalize_latents(
                conditioning.reference_latent.to(device=device),
                self.vae.latents_mean,
                self.vae.latents_std,
                self.vae.config.scaling_factor,
            ).to(base_tokens.dtype)
            ref_tokens = latent_ops.pack_latents(
                reference,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            )
            ref_coords = self.transformer.rope.prepare_video_coords(
                batch,
                reference.shape[2],
                reference.shape[3],
                reference.shape[4],
                device,
                fps=request_inputs.frame_rate,
            )
            ref_coords[:, 1:] *= conditioning.reference_downscale_factor
            latent_chunks.append(torch.zeros_like(ref_tokens))
            clean_chunks.append(ref_tokens)
            denoise_chunks.append(ref_tokens.new_zeros((*ref_tokens.shape[:2], 1)))
            coord_chunks.append(ref_coords)
            keyframe_mask_chunks.append(ref_tokens.new_zeros((*ref_tokens.shape[:2], 1)))

        initial = torch.cat(latent_chunks, dim=1)
        clean = torch.cat(clean_chunks, dim=1)
        denoise_mask = torch.cat(denoise_chunks, dim=1)
        latents = latent_ops.create_conditioned_noised_state(
            initial,
            clean,
            denoise_mask,
            noise_scale,
            request_inputs.generator,
        )
        return LTXPreparedVideoState(
            latents=latents,
            conditioning_mask=(1 - denoise_mask.squeeze(-1)),
            video_coords=torch.cat(coord_chunks, dim=2),
            keyframes_mask=torch.cat(keyframe_mask_chunks, dim=1),
            base_video_token_count=base_token_count,
            generated_keyframe_layout=generated_layout,
        )

    def _denoise_timestep_kwargs(
        self,
        ts: torch.Tensor,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
        *,
        video_token_count: int,
        audio_token_count: int,
    ) -> dict[str, torch.Tensor]:
        kwargs = super()._denoise_timestep_kwargs(
            ts,
            forward_ctx,
            denoise_ctx,
            video_token_count=video_token_count,
            audio_token_count=audio_token_count,
        )
        if getattr(self, "_active_phase_name", None) == "detail_fullres":
            kwargs["audio_timestep"] = torch.zeros_like(kwargs["audio_timestep"])
            kwargs["audio_sigma"] = torch.zeros_like(kwargs["audio_sigma"])
        return kwargs

    def _spatial_tile_indices(
        self,
        denoise_ctx: LTXDenoiseContext,
        forward_ctx: LTXForwardContext,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Build official 2x2 tile token indices and base-token blend masks."""
        base_count = denoise_ctx.base_video_token_count
        if base_count is None:
            raise RuntimeError("DFR tiled epilogue is missing its base token count.")
        frames = forward_ctx.latent_num_frames
        height = forward_ctx.latent_height
        width = forward_ctx.latent_width
        height_intervals = _split_two_tiles(height, _DFR_SPATIAL_OVERLAP)
        width_intervals = _split_two_tiles(width, _DFR_SPATIAL_OVERLAP)
        result = []
        for h_interval, w_interval in itertools.product(height_intervals, width_intervals):
            f = torch.arange(frames, device=denoise_ctx.latents.device)
            h = torch.arange(h_interval.start, h_interval.end, device=denoise_ctx.latents.device)
            w = torch.arange(w_interval.start, w_interval.end, device=denoise_ctx.latents.device)
            base_indices = (
                f[:, None, None] * height * width + h[None, :, None] * width + w[None, None, :]
            ).reshape(-1)
            keep_indices = base_indices
            if denoise_ctx.latents.shape[1] > base_count:
                tile_positions = denoise_ctx.video_coords[:, :, base_indices]
                tile_start = tile_positions[..., 0].amin(dim=2)
                tile_end = tile_positions[..., 1].amax(dim=2)
                cond_positions = denoise_ctx.video_coords[:, :, base_count:]
                overlaps = (cond_positions[..., 0] < tile_end[..., None]) & (
                    cond_positions[..., 1] > tile_start[..., None]
                )
                keep_cond = overlaps.all(dim=1).any(dim=0)
                cond_indices = base_count + keep_cond.nonzero(as_tuple=False).squeeze(1)
                keep_indices = torch.cat([base_indices, cond_indices])
            h_mask = _trapezoid(
                len(h),
                h_interval.left_ramp,
                h_interval.right_ramp,
                device=denoise_ctx.latents.device,
                dtype=denoise_ctx.latents.dtype,
            )
            w_mask = _trapezoid(
                len(w),
                w_interval.left_ramp,
                w_interval.right_ramp,
                device=denoise_ctx.latents.device,
                dtype=denoise_ctx.latents.dtype,
            )
            blend = (h_mask[:, None] * w_mask[None, :]).expand(frames, -1, -1).reshape(-1)
            result.append((keep_indices, blend))
        return result

    def _predict_noise_for_step(
        self,
        index: int,
        timestep: torch.Tensor,
        state: Any,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditioning = self._dfr_conditioning
        if conditioning is None or not conditioning.tiled:
            return super()._predict_noise_for_step(index, timestep, state, forward_ctx, denoise_ctx)

        base_count = denoise_ctx.base_video_token_count
        if base_count is None:
            raise RuntimeError("DFR tiled epilogue has no base token layout.")
        video_prediction = torch.zeros_like(state.video)
        audio_prediction = torch.zeros_like(state.audio)
        tiles = self._spatial_tile_indices(denoise_ctx, forward_ctx)
        for keep_indices, blend in tiles:
            local_coords = denoise_ctx.video_coords[:, :, keep_indices].clone()
            base_in_tile = blend.numel()
            offset = local_coords[:, :, :base_in_tile, :,].select(-1, 0).amin(dim=2, keepdim=True).unsqueeze(-1)
            local_coords -= offset
            local_mask = (
                None if denoise_ctx.conditioning_mask is None else denoise_ctx.conditioning_mask[:, keep_indices]
            )
            model_mask = denoise_ctx.conditioning_mask_for_model
            if model_mask is not None:
                model_mask = model_mask[:, keep_indices]
            local_ctx = replace(
                denoise_ctx,
                latents=state.video[:, keep_indices],
                video_coords=local_coords,
                conditioning_mask=local_mask,
                conditioning_mask_for_model=model_mask,
                keyframes_mask=(
                    None if denoise_ctx.keyframes_mask is None else denoise_ctx.keyframes_mask[:, keep_indices]
                ),
            )
            local_state = latent_ops.LTXAVState(video=state.video[:, keep_indices], audio=state.audio)
            local_video, local_audio = super()._predict_noise_for_step(
                index,
                timestep,
                local_state,
                forward_ctx,
                local_ctx,
            )
            base_indices = keep_indices[:base_in_tile]
            video_prediction[:, base_indices] += local_video[:, :base_in_tile] * blend[None, :, None]
            audio_prediction += local_audio
        audio_prediction /= len(tiles)
        return video_prediction, audio_prediction

    def _decode_lanczos_reencode_keyframes(
        self,
        keyframes: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None,
    ) -> torch.Tensor:
        if keyframes.ndim != 5:
            raise ValueError(f"DFR carry keyframes must be 5D, got {tuple(keyframes.shape)}.")
        encoded = []
        for index in range(keyframes.shape[2]):
            decoded = self.vae.decode(
                keyframes[:, :, index : index + 1].to(self.vae.dtype),
                None,
                return_dict=False,
            )[0]
            resized = torch.stack([_lanczos_x2_frame(frame[:, 0]) for frame in decoded], dim=0)
            resized = resized.to(device=self.device, dtype=self.vae.dtype).unsqueeze(2)
            if isinstance(generator, list):
                frame_generators = generator
            else:
                frame_generators = [generator] * resized.shape[0]
            raw = torch.cat(
                [
                    retrieve_latents(self.vae.encode(resized[item : item + 1]), frame_generators[item], "argmax")
                    for item in range(resized.shape[0])
                ],
                dim=0,
            )
            encoded.append(
                latent_ops.normalize_latents(
                    raw,
                    self.vae.latents_mean,
                    self.vae.latents_std,
                    self.vae.config.scaling_factor,
                )
            )
        return torch.cat(encoded, dim=2)

    def _run_dfr_phase(
        self,
        req: Any,
        inputs: LTXRequestInputs,
        phase: LTXPhaseRecipe,
        conditioning: _DFRVideoConditioning,
        *,
        sigmas: list[float] | None,
        image: Any | None,
        prompt_context: Any | None,
    ) -> LTXPhaseResult:
        self._enter_phase(phase)
        self._dfr_conditioning = conditioning
        try:
            return self.run_phase(
                req,
                inputs,
                noise_scale=phase.noise_scale if sigmas is None else float(sigmas[0]),
                sigmas=list(phase.sigmas) if sigmas is None and phase.sigmas is not None else sigmas,
                timesteps=None,
                attention_kwargs=None,
                phase_recipe=phase,
                image=image,
                prompt_context=prompt_context,
            )
        finally:
            self._dfr_conditioning = None

    def _run_recipe(
        self,
        req: Any,
        request_inputs: LTXRequestInputs,
        *,
        request_sigmas: list[float] | None,
        request_phase_sigmas: tuple[list[float] | None, ...] | None = None,
        image: Any | None = None,
    ) -> Any:
        if request_sigmas is not None:
            raise ValueError("LTX25DFRPipeline uses Stage-1 and detailing sigma schedules.")
        if request_inputs.frame_rate != 24.0:
            raise ValueError("The spatial-only LTX25DFRPipeline currently supports frame_rate=24 only.")

        requested_frames = request_inputs.num_frames
        canvas_frames, positions = resolve_dfr_canvas(requested_frames, self.vae_temporal_compression_ratio)
        phases = self.pipeline_recipe.phases
        phase_sigmas = request_phase_sigmas or (None,) * len(phases)
        prompt_context = None

        stage1_inputs = replace(
            request_inputs,
            height=request_inputs.height // 4,
            width=request_inputs.width // 4,
            num_frames=canvas_frames,
            num_inference_steps=phases[0].num_inference_steps or request_inputs.num_inference_steps,
            guidance=phases[0].guidance,
            latents=None,
            audio_latents=None,
        )
        stage1 = self._run_dfr_phase(
            req,
            stage1_inputs,
            phases[0],
            _DFRVideoConditioning(generated_positions=positions),
            sigmas=phase_sigmas[0],
            image=image,
            prompt_context=None,
        )
        prompt_context = stage1.forward_context.prompt_context
        if stage1.generated_keyframes is None or stage1.audio_for_next_phase is None:
            raise RuntimeError("DFR Stage 1 did not preserve generated keyframes and audio.")

        stage2_video = self._spatial_upsample_phase(stage1.video)
        stage2_slots = self._spatial_upsample_phase(stage1.generated_keyframes)
        stage2_inputs = replace(
            request_inputs,
            height=request_inputs.height // 2,
            width=request_inputs.width // 2,
            num_frames=canvas_frames,
            num_inference_steps=phases[1].num_inference_steps or request_inputs.num_inference_steps,
            guidance=phases[1].guidance,
            latents=stage2_video,
            audio_latents=stage1.audio_for_next_phase,
            audio_latents_normalized=True,
            decode_timestep=0.0,
            decode_noise_scale=None,
        )
        stage2 = self._run_dfr_phase(
            req,
            stage2_inputs,
            phases[1],
            _DFRVideoConditioning(
                generated_positions=positions,
                generated_initials=stage2_slots,
                reference_latent=stage1.video,
                reference_downscale_factor=getattr(self, "_dfr_reference_downscale", 1),
            ),
            sigmas=phase_sigmas[1],
            image=image,
            prompt_context=prompt_context,
        )
        if stage2.generated_keyframes is None:
            raise RuntimeError("DFR Stage 2 did not preserve generated keyframes.")

        carry_keyframes = self._decode_lanczos_reencode_keyframes(
            stage2.generated_keyframes,
            request_inputs.generator,
        )
        stage3_inputs = replace(
            request_inputs,
            num_frames=canvas_frames,
            num_inference_steps=phases[2].num_inference_steps or request_inputs.num_inference_steps,
            guidance=phases[2].guidance,
            latents=self._spatial_upsample_phase(stage2.video),
            audio_latents=stage1.audio_for_next_phase,
            audio_latents_normalized=True,
            decode_timestep=0.0,
            decode_noise_scale=None,
        )
        stage3 = self._run_dfr_phase(
            req,
            stage3_inputs,
            phases[2],
            _DFRVideoConditioning(
                reference_latent=stage2.video,
                reference_downscale_factor=getattr(self, "_dfr_reference_downscale", 1),
                encoded_keyframes=carry_keyframes,
                encoded_keyframe_positions=positions,
                tiled=True,
            ),
            sigmas=phase_sigmas[2],
            image=image,
            prompt_context=prompt_context,
        )

        keep_latent_frames = (requested_frames - 1) // self.vae_temporal_compression_ratio + 1
        final_phase = LTXPhaseResult(
            forward_context=replace(
                stage3.forward_context,
                request_inputs=replace(stage3.forward_context.request_inputs, num_frames=requested_frames),
            ),
            video=stage3.video[:, :, :keep_latent_frames],
            audio=stage1.audio,
        )
        output = self.decode_phase(final_phase)
        if not hasattr(output, "output") or not isinstance(output.output, tuple):
            return output
        video, audio = output.output
        if isinstance(audio, torch.Tensor) and audio.numel() and canvas_frames != requested_frames:
            keep_audio = round(audio.shape[-1] * requested_frames / canvas_frames)
            audio = audio[..., :keep_audio]
            output = replace(output, output=(video, audio))
        return output


# Keep post-processing discoverable beside the public pipeline entry.
from .ltx2_components import get_ltx2_post_process_func as get_ltx2_post_process_func  # noqa: E402,F401

