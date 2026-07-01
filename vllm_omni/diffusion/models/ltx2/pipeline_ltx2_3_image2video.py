# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LTX-2.3 image-to-video pipeline."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import PIL.Image
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import get_classifier_free_guidance_world_size
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch, split_diffusion_output_by_request

from .pipeline_ltx2 import calculate_shift
from .pipeline_ltx2_3 import LTX23Pipeline, get_ltx2_post_process_func
from .pipeline_ltx2_image2video import LTX2ImageToVideoPipeline, _I2VVideoAudioScheduler


class LTX23ImageToVideoPipeline(LTX23Pipeline):
    """LTX-2.3 image-to-video pipeline.

    This keeps the LTX-2.3 prompt connector, x0-space CFG, sigma prompt
    modulation, and audio branch semantics from ``LTX23Pipeline`` while
    reusing the existing LTX image-conditioning contract: the first video
    latent frame is encoded from the input image and remains fixed during
    denoising.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio, resample="bilinear")

    support_image_input = True

    _normalize_latents = staticmethod(LTX2ImageToVideoPipeline._normalize_latents)
    _create_noised_state = staticmethod(LTX2ImageToVideoPipeline._create_noised_state)

    @staticmethod
    def _resolve_single_prompt_image(raw_image: Any) -> Any:
        if isinstance(raw_image, list):
            if len(raw_image) != 1:
                raise ValueError(
                    "LTX-2.3 I2V prompt dictionaries support exactly one image per prompt. "
                    "Pass one image per prompt for batched I2V requests."
                )
            return raw_image[0]
        return raw_image

    @staticmethod
    def _resolve_additional_image(additional: dict[str, Any]) -> Any:
        raw_image = additional.get("preprocessed_image")
        if raw_image is None:
            raw_image = additional.get("pixel_values")
        if raw_image is None:
            raw_image = additional.get("image")
        return raw_image

    def prepare_latents(
        self,
        image: torch.Tensor | None = None,
        batch_size: int = 1,
        num_channels_latents: int = 128,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        noise_scale: float = 0.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare I2V latents and the first-frame conditioning mask.

        If caller-provided latents are used without an image, the latents must
        already represent the full video state including the conditioning first
        frame. Packed 3D latents are assumed to be in transformer token layout.
        """
        height = height // self.vae_spatial_compression_ratio
        width = width // self.vae_spatial_compression_ratio
        num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1

        shape = (batch_size, num_channels_latents, num_frames, height, width)
        mask_shape = (batch_size, 1, num_frames, height, width)

        if latents is not None:
            if latents.ndim == 5:
                batch_size, _, num_frames, height, width = latents.shape
                mask_shape = (batch_size, 1, num_frames, height, width)
                conditioning_mask = latents.new_zeros(mask_shape)
                conditioning_mask[:, :, 0] = 1.0

                latents = self._normalize_latents(
                    latents,
                    self.vae.latents_mean,
                    self.vae.latents_std,
                    self.vae.config.scaling_factor,
                )
                latents = self._create_noised_state(latents, noise_scale * (1 - conditioning_mask), generator)
                latents = self._pack_latents(
                    latents,
                    self.transformer_spatial_patch_size,
                    self.transformer_temporal_patch_size,
                )
            else:
                conditioning_mask = latents.new_zeros(mask_shape)
                conditioning_mask[:, :, 0] = 1.0

            conditioning_mask = self._pack_latents(
                conditioning_mask,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            ).squeeze(-1)
            if latents.ndim != 3 or latents.shape[:2] != conditioning_mask.shape:
                raise ValueError(
                    "Provided `latents` tensor has shape"
                    f" {latents.shape}, but the expected shape is {conditioning_mask.shape + (num_channels_latents,)}."
                )
            return latents.to(device=device, dtype=dtype), conditioning_mask

        if image is None:
            raise ValueError("`image` must be provided when `latents` is None.")

        image_batch_size = image.shape[0]
        if image_batch_size == 0:
            raise ValueError("`image` batch is empty.")
        if batch_size % image_batch_size != 0:
            raise ValueError(
                f"`batch_size` ({batch_size}) must be divisible by image batch size ({image_batch_size}) "
                "for image-to-video outputs."
            )
        num_videos_per_prompt = batch_size // image_batch_size

        if isinstance(generator, list):
            if len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective"
                    f" batch size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
            image_generators = [generator[i * num_videos_per_prompt] for i in range(image_batch_size)]
            init_latents = [
                retrieve_latents(self.vae.encode(image[i].unsqueeze(0).unsqueeze(2)), image_generators[i], "argmax")
                for i in range(image_batch_size)
            ]
        else:
            init_latents = [
                retrieve_latents(self.vae.encode(img.unsqueeze(0).unsqueeze(2)), generator, "argmax") for img in image
            ]

        init_latents = torch.cat(init_latents, dim=0).to(dtype)
        if num_videos_per_prompt > 1:
            init_latents = init_latents.repeat_interleave(num_videos_per_prompt, dim=0)
        init_latents = self._normalize_latents(
            init_latents,
            self.vae.latents_mean,
            self.vae.latents_std,
        )
        init_latents = init_latents.repeat(1, 1, num_frames, 1, 1)

        conditioning_mask = torch.zeros(mask_shape, device=device, dtype=dtype)
        conditioning_mask[:, :, 0] = 1.0

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = init_latents * conditioning_mask + noise * (1 - conditioning_mask)

        conditioning_mask = self._pack_latents(
            conditioning_mask,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        ).squeeze(-1)
        latents = self._pack_latents(latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size)

        return latents, conditioning_mask

    def check_inputs(
        self,
        image,
        height,
        width,
        prompt,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
    ):
        if image is None and latents is None:
            raise ValueError("Provide either `image` or `latents`. Cannot leave both undefined.")
        super().check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

    _step_video_latents_i2v = LTX2ImageToVideoPipeline._step_video_latents_i2v

    @torch.no_grad()
    def forward(
        self,
        req: DiffusionRequestBatch,
        image: PIL.Image.Image | torch.Tensor | list[PIL.Image.Image | torch.Tensor] | None = None,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        frame_rate: float | None = None,
        num_inference_steps: int | None = None,
        sigmas: list[float] | None = None,
        timesteps: list[int] | None = None,
        guidance_scale: float = 4.0,
        noise_scale: float = 0.0,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        decode_timestep: float | list[float] = 0.0,
        decode_noise_scale: float | list[float] | None = None,
        output_type: str = "np",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int | None = None,
    ) -> list[DiffusionOutput]:
        request_inputs = self._resolve_request_inputs(
            req,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            latents=latents,
            audio_latents=audio_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            output_type=output_type,
            max_sequence_length=max_sequence_length,
        )
        prompt = request_inputs.prompt
        negative_prompt = request_inputs.negative_prompt
        height = request_inputs.height
        width = request_inputs.width
        num_frames = request_inputs.num_frames
        frame_rate = request_inputs.frame_rate
        num_inference_steps = request_inputs.num_inference_steps
        guidance_scale = request_inputs.guidance_scale
        num_videos_per_prompt = request_inputs.num_videos_per_prompt
        generator = request_inputs.generator
        latents = request_inputs.latents
        audio_latents = request_inputs.audio_latents
        prompt_embeds = request_inputs.prompt_embeds
        negative_prompt_embeds = request_inputs.negative_prompt_embeds
        prompt_attention_mask = request_inputs.prompt_attention_mask
        negative_prompt_attention_mask = request_inputs.negative_prompt_attention_mask
        decode_timestep = request_inputs.decode_timestep
        decode_noise_scale = request_inputs.decode_noise_scale
        output_type = request_inputs.output_type
        max_sequence_length = request_inputs.max_sequence_length

        if image is None and req.prompts:
            raw_images = []
            for prompt_item in req.prompts:
                if isinstance(prompt_item, str):
                    raw_image = None
                else:
                    multi_modal_data = prompt_item.get("multi_modal_data") or {}
                    raw_image = multi_modal_data.get("image")
                    if raw_image is None:
                        additional = prompt_item.get("additional_information") or {}
                        raw_image = self._resolve_additional_image(additional)
                raw_image = self._resolve_single_prompt_image(raw_image)
                if isinstance(raw_image, str):
                    raw_image = PIL.Image.open(raw_image).convert("RGB")
                raw_images.append(raw_image)

            if any(img is None for img in raw_images):
                if latents is None:
                    raise ValueError("Image is required for LTX-2.3 I2V generation.")
            if len(raw_images) == 1:
                image = raw_images[0]
            elif raw_images:
                image = raw_images

        self.check_inputs(
            image=image,
            height=height,
            width=width,
            prompt=prompt,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None
        cfg_world_size = get_classifier_free_guidance_world_size()
        if self.do_classifier_free_guidance and cfg_world_size not in (1, 2):
            raise ValueError(
                f"LTX23Pipeline supports CFG parallelism with cfg_parallel_size 1 or 2, but got {cfg_world_size}."
            )
        cfg_parallel_ready = self.do_classifier_free_guidance and cfg_world_size > 1

        device = self.device
        prompt_context = self._prepare_prompt_context(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        batch_size = prompt_context.batch_size
        connector_prompt_embeds = prompt_context.connector_prompt_embeds
        connector_audio_prompt_embeds = prompt_context.connector_audio_prompt_embeds
        connector_attention_mask = prompt_context.connector_attention_mask
        positive_connector_prompt_embeds = prompt_context.positive_connector_prompt_embeds
        positive_connector_audio_prompt_embeds = prompt_context.positive_connector_audio_prompt_embeds
        positive_connector_attention_mask = prompt_context.positive_connector_attention_mask
        negative_connector_prompt_embeds = prompt_context.negative_connector_prompt_embeds
        negative_connector_audio_prompt_embeds = prompt_context.negative_connector_audio_prompt_embeds
        negative_connector_attention_mask = prompt_context.negative_connector_attention_mask

        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        if latents is not None:
            if latents.ndim == 5:
                _, _, latent_num_frames, latent_height, latent_width = latents.shape
            elif latents.ndim != 3:
                raise ValueError(
                    f"Provided `latents` tensor has shape {latents.shape}, but the expected shape is either "
                    "[batch_size, seq_len, num_features] or "
                    "[batch_size, latent_dim, latent_frames, latent_height, latent_width]."
                )

        if latents is None:
            if isinstance(image, torch.Tensor):
                if image.ndim == 3:
                    image = image.unsqueeze(0)
            elif isinstance(image, list) and image and isinstance(image[0], torch.Tensor):
                image = torch.stack(image, dim=0)
            else:
                image = self.video_processor.preprocess(image, height=height, width=width)
            image = image.to(device=device, dtype=positive_connector_prompt_embeds.dtype)

        num_channels_latents = self.transformer.config.in_channels
        latents, conditioning_mask = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            noise_scale,
            torch.float32,
            device,
            generator,
            latents,
        )

        duration_s = num_frames / frame_rate
        audio_latents_per_second = (
            self.audio_sampling_rate / self.audio_hop_length / float(self.audio_vae_temporal_compression_ratio)
        )
        audio_num_frames = round(duration_s * audio_latents_per_second)
        audio_num_frames = self._resolve_audio_latent_length(audio_num_frames, audio_latents)

        num_mel_bins = self.audio_vae.config.mel_bins if self.audio_vae is not None else 64
        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio
        num_channels_latents_audio = self.audio_vae.config.latent_channels if self.audio_vae is not None else 8
        audio_latents, original_audio_num_frames, padded_audio_num_frames = self.prepare_audio_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents_audio,
            audio_latent_length=audio_num_frames,
            num_mel_bins=num_mel_bins,
            noise_scale=noise_scale,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=audio_latents,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        mu = calculate_shift(
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_image_seq_len", 1024),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.95),
            self.scheduler.config.get("max_shift", 2.05),
        )
        audio_scheduler = copy.deepcopy(self.scheduler)
        video_audio_scheduler = _I2VVideoAudioScheduler(
            self,
            audio_scheduler,
            latent_num_frames,
            latent_height,
            latent_width,
        )
        _ = retrieve_timesteps(audio_scheduler, num_inference_steps, device, timesteps, sigmas=sigmas, mu=mu)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        self._num_timesteps = len(timesteps)

        video_coords = self.transformer.rope.prepare_video_coords(
            latents.shape[0],
            latent_num_frames,
            latent_height,
            latent_width,
            latents.device,
            fps=frame_rate,
        )
        audio_coords = self.transformer.audio_rope.prepare_audio_coords(
            audio_latents.shape[0],
            padded_audio_num_frames,
            audio_latents.device,
        )

        if self.do_classifier_free_guidance and not cfg_parallel_ready:
            video_coords = video_coords.repeat((2,) + (1,) * (video_coords.ndim - 1))
            audio_coords = audio_coords.repeat((2,) + (1,) * (audio_coords.ndim - 1))
            conditioning_mask_for_model = torch.cat([conditioning_mask, conditioning_mask])
        else:
            conditioning_mask_for_model = conditioning_mask

        with self.progress_bar(total=len(timesteps)) as pbar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if cfg_parallel_ready:
                    latent_model_input = latents.to(positive_connector_prompt_embeds.dtype)
                    audio_latent_model_input = audio_latents.to(positive_connector_prompt_embeds.dtype)
                    ts = t.expand(latent_model_input.shape[0])
                    video_ts = ts.unsqueeze(-1) * (1 - conditioning_mask)
                    positive_kwargs = {
                        "hidden_states": latent_model_input,
                        "audio_hidden_states": audio_latent_model_input,
                        "encoder_hidden_states": positive_connector_prompt_embeds,
                        "audio_encoder_hidden_states": positive_connector_audio_prompt_embeds,
                        "timestep": video_ts,
                        "audio_timestep": ts,
                        "sigma": ts,
                        "encoder_attention_mask": positive_connector_attention_mask,
                        "audio_encoder_attention_mask": positive_connector_attention_mask,
                        "num_frames": latent_num_frames,
                        "height": latent_height,
                        "width": latent_width,
                        "fps": frame_rate,
                        "audio_num_frames": padded_audio_num_frames,
                        "video_coords": video_coords,
                        "audio_coords": audio_coords,
                        "attention_kwargs": attention_kwargs,
                        "return_dict": False,
                    }
                    negative_kwargs = {
                        **positive_kwargs,
                        "encoder_hidden_states": negative_connector_prompt_embeds,
                        "audio_encoder_hidden_states": negative_connector_audio_prompt_embeds,
                        "encoder_attention_mask": negative_connector_attention_mask,
                        "audio_encoder_attention_mask": negative_connector_attention_mask,
                    }
                    noise_pred_video, noise_pred_audio = self.predict_noise_with_parallel_cfg(
                        true_cfg_scale=guidance_scale,
                        positive_kwargs=positive_kwargs,
                        negative_kwargs=negative_kwargs,
                        cfg_normalize=False,
                        video_latents=latents,
                        audio_latents=audio_latents,
                        video_sigma=self.scheduler.sigmas[i],
                        audio_sigma=audio_scheduler.sigmas[i],
                    )
                else:
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = latent_model_input.to(connector_prompt_embeds.dtype)
                    audio_latent_model_input = (
                        torch.cat([audio_latents] * 2) if self.do_classifier_free_guidance else audio_latents
                    )
                    audio_latent_model_input = audio_latent_model_input.to(connector_prompt_embeds.dtype)
                    ts = t.expand(latent_model_input.shape[0])
                    video_ts = ts.unsqueeze(-1) * (1 - conditioning_mask_for_model)

                    with self._transformer_cache_context("cond_uncond"):
                        noise_pred_video, noise_pred_audio = self.transformer(
                            hidden_states=latent_model_input,
                            audio_hidden_states=audio_latent_model_input,
                            encoder_hidden_states=connector_prompt_embeds,
                            audio_encoder_hidden_states=connector_audio_prompt_embeds,
                            timestep=video_ts,
                            audio_timestep=ts,
                            sigma=ts,
                            encoder_attention_mask=connector_attention_mask,
                            audio_encoder_attention_mask=connector_attention_mask,
                            num_frames=latent_num_frames,
                            height=latent_height,
                            width=latent_width,
                            fps=frame_rate,
                            audio_num_frames=padded_audio_num_frames,
                            video_coords=video_coords,
                            audio_coords=audio_coords,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )

                    noise_pred_video = noise_pred_video.float()
                    noise_pred_audio = noise_pred_audio.float()
                    if self.do_classifier_free_guidance:
                        noise_pred_video_uncond, noise_pred_video_cond = noise_pred_video.chunk(2)
                        noise_pred_video = self._combine_x0_space_cfg(
                            latents,
                            noise_pred_video_cond,
                            noise_pred_video_uncond,
                            self.scheduler.sigmas[i],
                            guidance_scale,
                        )

                        noise_pred_audio_uncond, noise_pred_audio_cond = noise_pred_audio.chunk(2)
                        noise_pred_audio = self._combine_x0_space_cfg(
                            audio_latents,
                            noise_pred_audio_cond,
                            noise_pred_audio_uncond,
                            audio_scheduler.sigmas[i],
                            guidance_scale,
                        )

                latents, audio_latents = self.scheduler_step_maybe_with_cfg(
                    (noise_pred_video, noise_pred_audio),
                    (t, t),
                    (latents, audio_latents),
                    do_true_cfg=self.do_classifier_free_guidance,
                    per_request_scheduler=video_audio_scheduler,
                )
                latents, audio_latents = self._synchronize_cfg_parallel_step_output(
                    (latents, audio_latents),
                    do_true_cfg=self.do_classifier_free_guidance,
                )
                pbar.update()

        latents = self._unpack_latents(
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )
        latents = self._denormalize_latents(
            latents,
            self.vae.latents_mean,
            self.vae.latents_std,
            self.vae.config.scaling_factor,
        )

        audio_latents = self._unpad_audio_latents(audio_latents, original_audio_num_frames)
        audio_latents = self._denormalize_audio_latents(
            audio_latents,
            self.audio_vae.latents_mean,
            self.audio_vae.latents_std,
        )
        audio_latents = self._unpack_audio_latents(
            audio_latents,
            original_audio_num_frames,
            num_mel_bins=latent_mel_bins,
        )

        return split_diffusion_output_by_request(
            self._decode_output(
                latents=latents,
                audio_latents=audio_latents,
                output_type=output_type,
                connector_prompt_embeds=connector_prompt_embeds,
                generator=generator,
                device=device,
                decode_timestep=decode_timestep,
                decode_noise_scale=decode_noise_scale,
                prompt_batch_size=batch_size,
            ),
            req,
            num_outputs_per_prompt=num_videos_per_prompt,
        )


__all__ = [
    "LTX23ImageToVideoPipeline",
    "get_ltx2_post_process_func",
]
