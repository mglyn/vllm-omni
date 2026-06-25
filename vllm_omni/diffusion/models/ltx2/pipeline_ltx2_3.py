# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Fully independent LTX-2.3 pipeline for vLLM-Omni.

This pipeline does NOT inherit from LTX2Pipeline because:
- LTX-2.3 connectors run per_token_rms_norm + per-modality video/audio
  projection internally (per_modality_projections=True),
  versus LTX-2's per_layer_masked_mean_norm + shared projection path
- LTX-2.3 uses a BWE vocoder outputting 48kHz audio (not 16kHz)
- LTX-2.3 transformer requires the sigma parameter for prompt modulation
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import PIL.Image
import torch
from diffusers import AutoencoderKLLTX2Audio, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2 import LTX2TextConnectors
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from huggingface_hub import hf_hub_download
from torch import nn
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import DistributedAutoencoderKLLTX2Video
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest

from .pipeline_ltx2 import (
    _get_prompt_field,
    _VideoAudioScheduler,
    calculate_shift,
    create_transformer_from_config,
    load_transformer_config,
)
from .pipeline_ltx2_image2video import LTX2ImageToVideoPipeline, _I2VVideoAudioScheduler

logger = init_logger(__name__)


def _is_output_rank() -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _vae_decode_needs_all_ranks(vae: Any) -> bool:
    if not torch.distributed.is_initialized():
        return False
    is_distributed_enabled = getattr(vae, "is_distributed_enabled", None)
    if not callable(is_distributed_enabled):
        return False
    try:
        return bool(is_distributed_enabled())
    except Exception:
        return False


def _should_decode_video_on_rank(vae: Any) -> bool:
    return _is_output_rank() or _vae_decode_needs_all_ranks(vae)


# Try to import LTX2VocoderWithBWE (diffusers >= 0.38.0)
try:
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
except ImportError:
    LTX2VocoderWithBWE = None


@dataclass
class _LTX23RequestInputs:
    prompt: str | list[str] | None
    negative_prompt: str | list[str] | None
    height: int
    width: int
    num_frames: int
    frame_rate: float
    num_inference_steps: int
    guidance_scale: float
    num_videos_per_prompt: int
    generator: torch.Generator | list[torch.Generator] | None
    latents: torch.Tensor | None
    audio_latents: torch.Tensor | None
    prompt_embeds: torch.Tensor | None
    negative_prompt_embeds: torch.Tensor | None
    prompt_attention_mask: torch.Tensor | None
    negative_prompt_attention_mask: torch.Tensor | None
    decode_timestep: float | list[float]
    decode_noise_scale: float | list[float] | None
    output_type: str
    max_sequence_length: int


@dataclass
class _LTX23PromptContext:
    batch_size: int
    connector_prompt_embeds: torch.Tensor
    connector_audio_prompt_embeds: torch.Tensor
    connector_attention_mask: torch.Tensor
    positive_connector_prompt_embeds: torch.Tensor
    positive_connector_audio_prompt_embeds: torch.Tensor
    positive_connector_attention_mask: torch.Tensor
    negative_connector_prompt_embeds: torch.Tensor | None
    negative_connector_audio_prompt_embeds: torch.Tensor | None
    negative_connector_attention_mask: torch.Tensor | None


def _stack_prompt_field_if_present(values: list[Any], field_name: str) -> torch.Tensor | None:
    if not any(value is not None for value in values):
        return None
    missing_indices = [idx for idx, value in enumerate(values) if value is None]
    if missing_indices:
        raise ValueError(
            f"`{field_name}` must be provided for every prompt when provided "
            f"for any prompt. Missing prompt indices: {missing_indices}."
        )
    return torch.stack(values)


def _detect_vocoder_output_sample_rate(model: str) -> int | None:
    """Detect the vocoder output sample rate from vocoder/config.json.

    This runs at factory time (engine process) so the rate is captured in
    the post-process closure and doesn't need cross-process communication.

    Returns:
        Output sample rate (e.g. 48000 for LTX-2.3 BWE vocoder) or None.
    """
    vocoder_config_path = os.path.join(model, "vocoder", "config.json")
    if not os.path.exists(vocoder_config_path):
        try:
            vocoder_config_path = hf_hub_download(model, "vocoder/config.json")
        except Exception:
            return None
    try:
        with open(vocoder_config_path) as f:
            cfg = json.load(f)
        return cfg.get("output_sampling_rate")
    except Exception:
        return None


def get_ltx2_post_process_func(od_config: OmniDiffusionConfig):
    """Factory for the LTX-2.3 post-process function.

    Detects the vocoder output sample rate at factory time and captures it
    in the closure so that the audio_sample_rate flows through
    DiffusionEngine -> OmniRequestOutput -> serving_video.
    """
    output_sr = _detect_vocoder_output_sample_rate(od_config.model)

    def post_process_func(output: tuple[torch.Tensor, torch.Tensor] | torch.Tensor):
        if isinstance(output, tuple) and len(output) == 2:
            video, audio = output
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu()
            result: dict[str, Any] = {"video": video, "audio": audio}
            if output_sr is not None:
                result["audio_sample_rate"] = output_sr
            return result
        return output

    return post_process_func


def _expand_per_prompt_decode_value(
    value: float | list[float],
    *,
    prompt_batch_size: int,
    effective_batch_size: int,
    field_name: str,
) -> list[float]:
    if not isinstance(value, list):
        return [value] * effective_batch_size
    if len(value) == 1:
        return value * effective_batch_size
    if len(value) == effective_batch_size:
        return value
    if prompt_batch_size > 0 and len(value) == prompt_batch_size and effective_batch_size % prompt_batch_size == 0:
        repeats = effective_batch_size // prompt_batch_size
        return [item for item in value for _ in range(repeats)]
    raise ValueError(
        f"`{field_name}` must have length 1, prompt batch size ({prompt_batch_size}), or effective batch size"
        f" ({effective_batch_size}); got {len(value)}."
    )


def _prepare_decode_timestep_conditioning(
    *,
    decode_timestep: float | list[float],
    decode_noise_scale: float | list[float] | None,
    prompt_batch_size: int,
    effective_batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    decode_timestep_values = _expand_per_prompt_decode_value(
        decode_timestep,
        prompt_batch_size=prompt_batch_size,
        effective_batch_size=effective_batch_size,
        field_name="decode_timestep",
    )
    if decode_noise_scale is None:
        decode_noise_scale_values = decode_timestep_values
    else:
        decode_noise_scale_values = _expand_per_prompt_decode_value(
            decode_noise_scale,
            prompt_batch_size=prompt_batch_size,
            effective_batch_size=effective_batch_size,
            field_name="decode_noise_scale",
        )
    return (
        torch.tensor(decode_timestep_values, device=device, dtype=dtype),
        torch.tensor(decode_noise_scale_values, device=device, dtype=dtype)[:, None, None, None, None],
    )


class LTX23Pipeline(
    nn.Module,
    CFGParallelMixin,
    ProgressBarMixin,
    SupportsComponentDiscovery,
    DiffusionPipelineProfilerMixin,
):
    """Fully independent LTX-2.3 pipeline.

    Key differences from LTX2Pipeline:
    - Text encoding: uses ALL 49 hidden states from Gemma-3-12B, flattened
    - Connectors: uses padding_side API (not additive_mask)
    - Vocoder: uses LTX2VocoderWithBWE (48kHz output)
    - Transformer: passes sigma for prompt_adaln
    """

    # Audio is diffused jointly with video; warmup must size audio tokens.
    dummy_run_num_frames = 2
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "connectors"]
    _vae_modules: ClassVar[list[str]] = ["vae", "audio_vae"]
    _resident_modules: ClassVar[list[str]] = ["vocoder"]

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        model = od_config.model
        local_files_only = os.path.exists(model)

        # Weight sources for transformer (loaded via AutoWeightsLoader)
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        # See ``hub_prefetch.py`` for the transformers v5 multi-worker subfolder
        # race; prefetch the whole component set before any from_pretrained.
        ltx2_subfolders = [
            "tokenizer",
            "text_encoder",
            "connectors",
            "vae",
            "audio_vae",
            "vocoder",
            "scheduler",
        ]
        prefetch_subfolders(model, ltx2_subfolders, local_files_only=local_files_only)

        # --- Tokenizer (lightweight, stays wherever) ---
        self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)

        # --- Text encoder ---
        with torch.device("cpu"):
            self.text_encoder = from_pretrained_with_prefetch(
                Gemma3ForConditionalGeneration.from_pretrained,
                model,
                subfolder="text_encoder",
                prefetch_list=ltx2_subfolders,
                local_files_only=local_files_only,
                torch_dtype=dtype,
            )

        # --- Connectors (LTX-2.3 connectors include caption projection) ---
        self.connectors = from_pretrained_with_prefetch(
            LTX2TextConnectors.from_pretrained,
            model,
            subfolder="connectors",
            prefetch_list=ltx2_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        )

        # --- VAE, Audio VAE ---
        self.vae = from_pretrained_with_prefetch(
            DistributedAutoencoderKLLTX2Video.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=ltx2_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        )
        self.audio_vae = from_pretrained_with_prefetch(
            AutoencoderKLLTX2Audio.from_pretrained,
            model,
            subfolder="audio_vae",
            prefetch_list=ltx2_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        )

        # --- Vocoder: prefer BWE vocoder (48kHz) for LTX-2.3 ---
        vocoder_cls = LTX2VocoderWithBWE or LTX2Vocoder
        try:
            self.vocoder = vocoder_cls.from_pretrained(
                model, subfolder="vocoder", torch_dtype=dtype, local_files_only=local_files_only
            )
        except (TypeError, OSError, ValueError):
            self.vocoder = LTX2Vocoder.from_pretrained(
                model, subfolder="vocoder", torch_dtype=dtype, local_files_only=local_files_only
            )

        # --- Transformer: created empty, weights loaded via AutoWeightsLoader ---
        transformer_config = load_transformer_config(model, "transformer", local_files_only)
        quant_config = getattr(self.od_config, "quantization_config", None)
        self.transformer = create_transformer_from_config(transformer_config, quant_config=quant_config)
        self._place_aux_components()

        # --- Scheduler ---
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )

        # --- Derived compression ratios ---
        self.vae_spatial_compression_ratio = self.vae.spatial_compression_ratio if self.vae is not None else 32
        self.vae_temporal_compression_ratio = self.vae.temporal_compression_ratio if self.vae is not None else 8
        self.audio_vae_mel_compression_ratio = self.audio_vae.mel_compression_ratio if self.audio_vae is not None else 4
        self.audio_vae_temporal_compression_ratio = (
            self.audio_vae.temporal_compression_ratio if self.audio_vae is not None else 4
        )
        self.transformer_spatial_patch_size = self.transformer.config.patch_size if self.transformer is not None else 1
        self.transformer_temporal_patch_size = (
            self.transformer.config.patch_size_t if self.transformer is not None else 1
        )
        self.audio_sampling_rate = self.audio_vae.config.sample_rate if self.audio_vae is not None else 16000
        self.audio_hop_length = self.audio_vae.config.mel_hop_length if self.audio_vae is not None else 160

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)

        # Tokenizer max length
        tokenizer_max_length = 1024
        if self.tokenizer is not None:
            tokenizer_max_length = self.tokenizer.model_max_length
            if tokenizer_max_length is None or tokenizer_max_length > 100000:
                encoder_config = getattr(self.text_encoder, "config", None)
                config_max_len = getattr(encoder_config, "max_position_embeddings", None)
                if config_max_len is None:
                    config_max_len = getattr(encoder_config, "max_seq_len", None)
                tokenizer_max_length = config_max_len or 1024
        self.tokenizer_max_length = int(tokenizer_max_length)

        # Pipeline state
        self._guidance_scale = None
        self._attention_kwargs = None
        self._interrupt = False
        self._num_timesteps = None
        self._current_timestep = None

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def _place_aux_components(self) -> None:
        parallel_config = getattr(self.od_config, "parallel_config", None)
        use_managed_placement = bool(
            getattr(self.od_config, "enable_cpu_offload", False)
            or getattr(self.od_config, "enable_layerwise_offload", False)
            or getattr(parallel_config, "use_hsdp", False)
        )
        if use_managed_placement:
            return

        modules = ModuleDiscovery.discover(self)
        for module in (*modules.encoders, *modules.vaes, *modules.resident_modules):
            module.to(self.device)

    # ------------------------------------------------------------------
    # Text Encoding (LTX-2.3 specific)
    # ------------------------------------------------------------------

    def _get_gemma_prompt_embeds(
        self,
        prompt: str | list[str],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 1024,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Encode prompts using Gemma-3-12B, returning ALL 49 hidden states flattened.

        Stacks all 49 hidden states and flattens to [B, seq, hidden * 49]. The
        connectors unflatten, apply per_token_rms_norm, and project internally
        (same shape contract as LTX-2 since the `diffusers==0.38` connector
        migration; the two differ only in the connector's internal norm path).
        """
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.tokenizer is not None:
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        prompt = [p.strip() for p in prompt]
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_attention_mask = text_inputs.attention_mask.to(device)

        text_encoder_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        )

        hidden_states = text_encoder_outputs.hidden_states

        # LTX-2.3: Stack ALL 49 hidden states and flatten
        # [49 x (B, seq, 3840)] -> [B, seq, 3840, 49] -> [B, seq, 188160]
        prompt_embeds = torch.stack(hidden_states, dim=-1).flatten(2, 3).to(dtype=dtype)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self.device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type as `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            if isinstance(negative_prompt, list) and batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    # ------------------------------------------------------------------
    # Latent utilities (shared with LTX2Pipeline)
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            batch_size,
            -1,
            post_patch_num_frames,
            patch_size_t,
            post_patch_height,
            patch_size,
            post_patch_width,
            patch_size,
        )
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents

    @staticmethod
    def _unpack_latents(
        latents: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
    ) -> torch.Tensor:
        batch_size = latents.size(0)
        latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return latents

    @staticmethod
    def _normalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * scaling_factor / latents_std
        return latents

    @staticmethod
    def _normalize_audio_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor):
        latents_mean = latents_mean.to(latents.device, latents.dtype)
        latents_std = latents_std.to(latents.device, latents.dtype)
        return (latents - latents_mean) / latents_std

    @staticmethod
    def _denormalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / scaling_factor + latents_mean
        return latents

    @staticmethod
    def _denormalize_audio_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor):
        latents_mean = latents_mean.to(latents.device, latents.dtype)
        latents_std = latents_std.to(latents.device, latents.dtype)
        return (latents * latents_std) + latents_mean

    @staticmethod
    def _pack_audio_latents(
        latents: torch.Tensor, patch_size: int | None = None, patch_size_t: int | None = None
    ) -> torch.Tensor:
        if patch_size is not None and patch_size_t is not None:
            batch_size, num_channels, latent_length, latent_mel_bins = latents.shape
            post_patch_latent_length = latent_length / patch_size_t
            post_patch_mel_bins = latent_mel_bins / patch_size
            latents = latents.reshape(
                batch_size, -1, post_patch_latent_length, patch_size_t, post_patch_mel_bins, patch_size
            )
            latents = latents.permute(0, 2, 4, 1, 3, 5).flatten(3, 5).flatten(1, 2)
        else:
            latents = latents.transpose(1, 2).flatten(2, 3)
        return latents

    @staticmethod
    def _unpack_audio_latents(
        latents: torch.Tensor,
        latent_length: int,
        num_mel_bins: int,
        patch_size: int | None = None,
        patch_size_t: int | None = None,
    ) -> torch.Tensor:
        if patch_size is not None and patch_size_t is not None:
            batch_size = latents.size(0)
            latents = latents.reshape(batch_size, latent_length, num_mel_bins, -1, patch_size_t, patch_size)
            latents = latents.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)
        else:
            latents = latents.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)
        return latents

    @staticmethod
    def _unpad_audio_latents(latents: torch.Tensor, num_frames: int) -> torch.Tensor:
        return latents[:, :num_frames]

    @staticmethod
    def _get_sp_padded_audio_latent_length(audio_latent_length: int, sp_size: int) -> int:
        if sp_size > 1:
            audio_latent_length += (sp_size - (audio_latent_length % sp_size)) % sp_size
        return audio_latent_length

    def _resolve_audio_latent_length(self, audio_latent_length: int, audio_latents: torch.Tensor | None) -> int:
        if audio_latents is None or audio_latents.ndim != 4:
            return audio_latent_length

        provided_latent_length = audio_latents.shape[2]
        sp_size = getattr(self.od_config.parallel_config, "sequence_parallel_size", 1) or 1
        padded_latent_length = self._get_sp_padded_audio_latent_length(audio_latent_length, int(sp_size))

        # Keep requested duration semantics when callers pass 4D latents that
        # are already padded for SP; other 4D lengths retain shape inference.
        if provided_latent_length in {audio_latent_length, padded_latent_length}:
            return audio_latent_length
        return provided_latent_length

    def _decode_output(
        self,
        *,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
        output_type: str,
        connector_prompt_embeds: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None,
        device: torch.device,
        decode_timestep: float | list[float],
        decode_noise_scale: float | list[float] | None,
        prompt_batch_size: int,
    ) -> DiffusionOutput:
        if output_type == "latent":
            return DiffusionOutput(
                output=(latents, audio_latents),
                stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            )

        latents = latents.to(connector_prompt_embeds.dtype)
        if not self.vae.config.timestep_conditioning:
            timestep_decode = None
        else:
            noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
            timestep_decode, decode_noise_scale_t = _prepare_decode_timestep_conditioning(
                decode_timestep=decode_timestep,
                decode_noise_scale=decode_noise_scale,
                prompt_batch_size=prompt_batch_size,
                effective_batch_size=latents.shape[0],
                device=device,
                dtype=latents.dtype,
            )
            latents = (1 - decode_noise_scale_t) * latents + decode_noise_scale_t * noise

        if _should_decode_video_on_rank(self.vae):
            latents = latents.to(self.vae.dtype)
            video = self.vae.decode(latents, timestep_decode, return_dict=False)[0]
        else:
            video = torch.empty(0, device=latents.device, dtype=latents.dtype)

        if not _is_output_rank():
            return DiffusionOutput(
                output=(
                    torch.empty(0, device=video.device, dtype=video.dtype),
                    torch.empty(0, device=audio_latents.device, dtype=audio_latents.dtype),
                ),
                stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            )

        if video.numel() > 0:
            video = self.video_processor.postprocess_video(video, output_type=output_type)

        audio_latents = audio_latents.to(self.audio_vae.dtype)
        generated_mel_spectrograms = self.audio_vae.decode(audio_latents, return_dict=False)[0]
        audio = self.vocoder(generated_mel_spectrograms)

        return DiffusionOutput(
            output=(video, audio),
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    # ------------------------------------------------------------------
    # Latent preparation
    # ------------------------------------------------------------------

    def prepare_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 128,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        noise_scale: float = 0.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            if latents.ndim == 5:
                latents = self._normalize_latents(
                    latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
                )
                latents = self._pack_latents(
                    latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
                )
            if latents.ndim != 3:
                raise ValueError(f"Provided `latents` has shape {latents.shape}, expected [batch, seq, features].")
            noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
            latents = noise_scale * noise + (1 - noise_scale) * latents
            return latents.to(device=device, dtype=dtype)

        height = height // self.vae_spatial_compression_ratio
        width = width // self.vae_spatial_compression_ratio
        num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        shape = (batch_size, num_channels_latents, num_frames, height, width)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size)
        return latents

    def prepare_audio_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 8,
        audio_latent_length: int = 1,
        num_mel_bins: int = 64,
        noise_scale: float = 0.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int, int]:
        original_latent_length = audio_latent_length
        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio

        sp_size = getattr(self.od_config.parallel_config, "sequence_parallel_size", 1) or 1
        padded_latent_length = self._get_sp_padded_audio_latent_length(original_latent_length, int(sp_size))

        if latents is not None:
            if latents.ndim == 4:
                latents = self._pack_audio_latents(latents)
            if latents.ndim != 3:
                raise ValueError(f"Provided `latents` has shape {latents.shape}, expected [batch, seq, features].")
            latents = self._normalize_audio_latents(latents, self.audio_vae.latents_mean, self.audio_vae.latents_std)
            noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
            latents = noise_scale * noise + (1 - noise_scale) * latents

            if latents.shape[1] not in {original_latent_length, padded_latent_length}:
                raise ValueError(
                    "Provided `audio_latents` has incompatible audio frame count "
                    f"{latents.shape[1]}; expected {original_latent_length} or {padded_latent_length}."
                )

            if latents.shape[1] == original_latent_length and padded_latent_length > original_latent_length:
                padding = torch.zeros(
                    latents.shape[0],
                    padded_latent_length - original_latent_length,
                    latents.shape[2],
                    dtype=latents.dtype,
                    device=latents.device,
                )
                latents = torch.cat([latents, padding], dim=1)

            return latents.to(device=device, dtype=dtype), original_latent_length, padded_latent_length

        shape = (batch_size, num_channels_latents, padded_latent_length, latent_mel_bins)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_audio_latents(latents)
        return latents, original_latent_length, padded_latent_length

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
    ):
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 32 but are {height} and {width}.")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`.")
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        elif prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when "
                    "passed directly, but got: `prompt_attention_mask` "
                    f"{prompt_attention_mask.shape} != `negative_prompt_attention_mask` "
                    f"{negative_prompt_attention_mask.shape}."
                )

    # ------------------------------------------------------------------
    # Cache context
    # ------------------------------------------------------------------

    def _transformer_cache_context(self, context_name: str):
        cache_context = getattr(self.transformer, "cache_context", None)
        if callable(cache_context):
            return cache_context(context_name)
        return nullcontext()

    # ------------------------------------------------------------------
    # CFG helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _combine_x0_space_cfg(
        sample: torch.Tensor,
        positive_noise_pred: torch.Tensor,
        negative_noise_pred: torch.Tensor,
        sigma: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        x0_cond = sample - positive_noise_pred * sigma
        x0_uncond = sample - negative_noise_pred * sigma
        x0_guided = x0_cond + (guidance_scale - 1) * (x0_cond - x0_uncond)
        return (sample - x0_guided) / sigma

    def predict_noise(self, **kwargs):
        with self._transformer_cache_context("cond_uncond"):
            noise_pred_video, noise_pred_audio = self.transformer(**kwargs)
        return noise_pred_video.float(), noise_pred_audio.float()

    def combine_cfg_noise(
        self,
        positive_noise_pred,
        negative_noise_pred,
        true_cfg_scale,
        cfg_normalize=False,
        *,
        video_latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        video_sigma: torch.Tensor | None = None,
        audio_sigma: torch.Tensor | None = None,
    ):
        if video_latents is None or audio_latents is None or video_sigma is None or audio_sigma is None:
            raise ValueError("LTX23Pipeline applies CFG in x0-space and requires video/audio latents and sigmas.")

        video_pos, audio_pos = positive_noise_pred
        video_neg, audio_neg = negative_noise_pred
        video_combined = self._combine_x0_space_cfg(
            video_latents,
            video_pos,
            video_neg,
            video_sigma,
            true_cfg_scale,
        )
        audio_combined = self._combine_x0_space_cfg(
            audio_latents,
            audio_pos,
            audio_neg,
            audio_sigma,
            true_cfg_scale,
        )
        if cfg_normalize:
            video_combined = self.cfg_normalize_function(video_pos, video_combined)
            audio_combined = self.cfg_normalize_function(audio_pos, audio_combined)
        return video_combined, audio_combined

    def predict_noise_with_parallel_cfg(
        self,
        true_cfg_scale: float,
        positive_kwargs: dict[str, Any],
        negative_kwargs: dict[str, Any],
        cfg_normalize: bool = True,
        output_slice: int | None = None,
        *,
        video_latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        video_sigma: torch.Tensor | None = None,
        audio_sigma: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def maybe_slice(pred: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            if output_slice is None:
                return pred
            return pred[0][:, :output_slice], pred[1][:, :output_slice]

        cfg_world_size = get_classifier_free_guidance_world_size()
        if cfg_world_size != 2:
            raise ValueError(f"LTX23Pipeline parallel CFG requires cfg_parallel_size 2, but got {cfg_world_size}.")

        cfg_group = get_cfg_group()
        cfg_rank = get_classifier_free_guidance_rank()
        branch_kwargs = positive_kwargs if cfg_rank == 0 else negative_kwargs
        local_video_pred, local_audio_pred = maybe_slice(self.predict_noise(**branch_kwargs))

        gathered_video = cfg_group.all_gather(local_video_pred, separate_tensors=True)
        gathered_audio = cfg_group.all_gather(local_audio_pred, separate_tensors=True)
        positive_noise_pred = (gathered_video[0], gathered_audio[0])
        negative_noise_pred = (gathered_video[1], gathered_audio[1])

        return self.combine_cfg_noise(
            positive_noise_pred,
            negative_noise_pred,
            true_cfg_scale,
            cfg_normalize,
            video_latents=video_latents,
            audio_latents=audio_latents,
            video_sigma=video_sigma,
            audio_sigma=audio_sigma,
        )

    def _synchronize_cfg_parallel_step_output(
        self,
        latents: tuple[torch.Tensor, torch.Tensor],
        do_true_cfg: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (do_true_cfg and get_classifier_free_guidance_world_size() > 1):
            return latents

        latents = tuple(tensor.contiguous() for tensor in latents)
        device = next((tensor.device for tensor in latents if tensor.is_cuda), None)
        if device is not None:
            torch.cuda.current_stream(device).synchronize()
        return latents

    def _resolve_request_inputs(
        self,
        req: OmniDiffusionRequest,
        *,
        prompt: str | list[str] | None,
        negative_prompt: str | list[str] | None,
        height: int | None,
        width: int | None,
        num_frames: int | None,
        frame_rate: float | None,
        num_inference_steps: int | None,
        timesteps: list[int] | None,
        guidance_scale: float,
        num_videos_per_prompt: int | None,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None,
        audio_latents: torch.Tensor | None,
        prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds: torch.Tensor | None,
        prompt_attention_mask: torch.Tensor | None,
        negative_prompt_attention_mask: torch.Tensor | None,
        decode_timestep: float | list[float],
        decode_noise_scale: float | list[float] | None,
        output_type: str,
        max_sequence_length: int | None,
    ) -> _LTX23RequestInputs:
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts] or prompt
        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in req.prompts):
            negative_prompt = None
        elif req.prompts:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in req.prompts]

        height = req.sampling_params.height or height or 512
        width = req.sampling_params.width or width or 768
        num_frames = req.sampling_params.num_frames or num_frames or 121
        frame_rate = req.sampling_params.resolved_frame_rate or frame_rate or 24.0
        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps or 40
        if timesteps is None:
            num_inference_steps = max(int(num_inference_steps), 2)
        elif len(timesteps) < 2:
            raise ValueError("`timesteps` must contain at least 2 values for FlowMatchEulerDiscreteScheduler.")

        num_videos_per_prompt = (
            req.sampling_params.num_outputs_per_prompt
            if req.sampling_params.num_outputs_per_prompt > 0
            else num_videos_per_prompt or 1
        )
        max_sequence_length = (
            req.sampling_params.max_sequence_length or max_sequence_length or self.tokenizer_max_length
        )

        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale

        if generator is None:
            generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(req.sampling_params.seed)

        latents = req.sampling_params.latents if req.sampling_params.latents is not None else latents
        audio_latents = (
            req.sampling_params.audio_latents
            if req.sampling_params.audio_latents is not None
            else req.sampling_params.extra_args.get("audio_latents", audio_latents)
        )

        req_prompt_embeds = [_get_prompt_field(p, "prompt_embeds") for p in req.prompts]
        stacked_prompt_embeds = _stack_prompt_field_if_present(req_prompt_embeds, "prompt_embeds")
        if stacked_prompt_embeds is not None:
            prompt_embeds = stacked_prompt_embeds
            prompt = None

        req_negative_prompt_embeds = [_get_prompt_field(p, "negative_prompt_embeds") for p in req.prompts]
        stacked_negative_prompt_embeds = _stack_prompt_field_if_present(
            req_negative_prompt_embeds, "negative_prompt_embeds"
        )
        if stacked_negative_prompt_embeds is not None:
            negative_prompt_embeds = stacked_negative_prompt_embeds
            negative_prompt = None

        req_prompt_attention_masks = []
        for prompt_item in req.prompts:
            mask = _get_prompt_field(prompt_item, "prompt_attention_mask")
            if mask is None:
                mask = _get_prompt_field(prompt_item, "attention_mask")
            req_prompt_attention_masks.append(mask)
        stacked_prompt_attention_mask = _stack_prompt_field_if_present(
            req_prompt_attention_masks, "prompt_attention_mask"
        )
        if stacked_prompt_attention_mask is not None:
            prompt_attention_mask = stacked_prompt_attention_mask

        req_negative_attention_masks = []
        for prompt_item in req.prompts:
            mask = _get_prompt_field(prompt_item, "negative_prompt_attention_mask")
            if mask is None:
                mask = _get_prompt_field(prompt_item, "negative_attention_mask")
            req_negative_attention_masks.append(mask)
        stacked_negative_prompt_attention_mask = _stack_prompt_field_if_present(
            req_negative_attention_masks, "negative_prompt_attention_mask"
        )
        if stacked_negative_prompt_attention_mask is not None:
            negative_prompt_attention_mask = stacked_negative_prompt_attention_mask

        if req.sampling_params.decode_timestep is not None:
            decode_timestep = req.sampling_params.decode_timestep
        if req.sampling_params.decode_noise_scale is not None:
            decode_noise_scale = req.sampling_params.decode_noise_scale
        if req.sampling_params.output_type is not None:
            output_type = req.sampling_params.output_type

        return _LTX23RequestInputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            frame_rate=float(frame_rate),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=guidance_scale,
            num_videos_per_prompt=int(num_videos_per_prompt),
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
            max_sequence_length=int(max_sequence_length),
        )

    def _prepare_prompt_context(
        self,
        *,
        prompt: str | list[str] | None,
        negative_prompt: str | list[str] | None,
        prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds: torch.Tensor | None,
        prompt_attention_mask: torch.Tensor | None,
        negative_prompt_attention_mask: torch.Tensor | None,
        num_videos_per_prompt: int,
        max_sequence_length: int,
    ) -> _LTX23PromptContext:
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
            self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                max_sequence_length=max_sequence_length,
                device=self.device,
            )
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = self.connectors(
            prompt_embeds,
            prompt_attention_mask,
            padding_side=getattr(self.tokenizer, "padding_side", "left"),
        )

        positive_connector_prompt_embeds = connector_prompt_embeds
        positive_connector_audio_prompt_embeds = connector_audio_prompt_embeds
        positive_connector_attention_mask = connector_attention_mask
        negative_connector_prompt_embeds = None
        negative_connector_audio_prompt_embeds = None
        negative_connector_attention_mask = None
        if self.do_classifier_free_guidance:
            split_batch = batch_size * num_videos_per_prompt
            negative_connector_prompt_embeds = connector_prompt_embeds[:split_batch]
            positive_connector_prompt_embeds = connector_prompt_embeds[split_batch:]
            negative_connector_audio_prompt_embeds = connector_audio_prompt_embeds[:split_batch]
            positive_connector_audio_prompt_embeds = connector_audio_prompt_embeds[split_batch:]
            negative_connector_attention_mask = connector_attention_mask[:split_batch]
            positive_connector_attention_mask = connector_attention_mask[split_batch:]

        return _LTX23PromptContext(
            batch_size=batch_size,
            connector_prompt_embeds=connector_prompt_embeds,
            connector_audio_prompt_embeds=connector_audio_prompt_embeds,
            connector_attention_mask=connector_attention_mask,
            positive_connector_prompt_embeds=positive_connector_prompt_embeds,
            positive_connector_audio_prompt_embeds=positive_connector_audio_prompt_embeds,
            positive_connector_attention_mask=positive_connector_attention_mask,
            negative_connector_prompt_embeds=negative_connector_prompt_embeds,
            negative_connector_audio_prompt_embeds=negative_connector_audio_prompt_embeds,
            negative_connector_attention_mask=negative_connector_attention_mask,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        req: OmniDiffusionRequest,
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
    ) -> DiffusionOutput:
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

        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
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

        # ---- Prepare latents ----
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        if latents is not None and latents.ndim == 5:
            _, _, latent_num_frames, latent_height, latent_width = latents.shape

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
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

        # ---- Scheduler setup ----
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        # Use max_image_seq_len (not actual video_sequence_length) for mu calculation,
        # matching diffusers' LTX2Pipeline which hardcodes this value.
        mu = calculate_shift(
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_image_seq_len", 1024),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.95),
            self.scheduler.config.get("max_shift", 2.05),
        )
        audio_scheduler = copy.deepcopy(self.scheduler)
        video_audio_scheduler = _VideoAudioScheduler(self.scheduler, audio_scheduler)
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

        # ---- RoPE coordinates ----
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

        # ---- CFG: duplicate coords for single-rank batch=2 CFG ----
        # Connector outputs are already batch=2 (neg+pos concatenated before connector call)
        if self.do_classifier_free_guidance and not cfg_parallel_ready:
            video_coords = video_coords.repeat((2,) + (1,) * (video_coords.ndim - 1))
            audio_coords = audio_coords.repeat((2,) + (1,) * (audio_coords.ndim - 1))

        # ---- Denoising loop ----
        # Uses x0-space CFG (delta formulation) matching diffusers' LTX2Pipeline.
        # The velocity predictions are converted to x0, guidance is applied in x0
        # space, then converted back to velocity for the scheduler step.
        with self.progress_bar(total=len(timesteps)) as pbar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if cfg_parallel_ready:
                    latent_model_input = latents.to(positive_connector_prompt_embeds.dtype)
                    audio_latent_model_input = audio_latents.to(positive_connector_prompt_embeds.dtype)
                    ts = t.expand(latent_model_input.shape[0])
                    positive_kwargs = {
                        "hidden_states": latent_model_input,
                        "audio_hidden_states": audio_latent_model_input,
                        "encoder_hidden_states": positive_connector_prompt_embeds,
                        "audio_encoder_hidden_states": positive_connector_audio_prompt_embeds,
                        "timestep": ts,
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
                else:
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = latent_model_input.to(connector_prompt_embeds.dtype)
                    audio_latent_model_input = (
                        torch.cat([audio_latents] * 2) if self.do_classifier_free_guidance else audio_latents
                    )
                    audio_latent_model_input = audio_latent_model_input.to(connector_prompt_embeds.dtype)
                    ts = t.expand(latent_model_input.shape[0])

                    with self._transformer_cache_context("cond_uncond"):
                        noise_pred_video, noise_pred_audio = self.transformer(
                            hidden_states=latent_model_input,
                            audio_hidden_states=audio_latent_model_input,
                            encoder_hidden_states=connector_prompt_embeds,
                            audio_encoder_hidden_states=connector_audio_prompt_embeds,
                            timestep=ts,
                            sigma=ts,  # LTX-2.3: sigma for prompt_adaln
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

                    latents = self.scheduler.step(noise_pred_video, t, latents, return_dict=False)[0]
                    audio_latents = audio_scheduler.step(noise_pred_audio, t, audio_latents, return_dict=False)[0]

                pbar.update()

        # ---- Unpack and denormalize ----
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

        # ---- Decode ----
        return self._decode_output(
            latents=latents,
            audio_latents=audio_latents,
            output_type=output_type,
            connector_prompt_embeds=connector_prompt_embeds,
            generator=generator,
            device=device,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            prompt_batch_size=batch_size,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


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
        req: OmniDiffusionRequest,
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
    ) -> DiffusionOutput:
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

        return self._decode_output(
            latents=latents,
            audio_latents=audio_latents,
            output_type=output_type,
            connector_prompt_embeds=connector_prompt_embeds,
            generator=generator,
            device=device,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            prompt_batch_size=batch_size,
        )
