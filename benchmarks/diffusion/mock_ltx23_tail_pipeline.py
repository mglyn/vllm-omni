# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mock LTX-2.3 tail pipeline for video VAE/postprocess/audio timing.

This intentionally does not load tokenizer, text encoder, connectors, or the
diffusion transformer. It starts from random post-denoise latents and measures:

    video VAE decode -> video postprocess
    audio VAE decode -> vocoder

Use it to isolate the tail optimization on machines that cannot hold the full
LTX-2.3 pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from torch.profiler import ProfilerActivity, profile, record_function

from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import (
    LTX23_ASYNC_VIDEO_POSTPROCESS_ENV,
    _LTX23VideoPostprocessor,
)
from vllm_omni.platforms import current_omni_platform

try:
    from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder, LTX2VocoderWithBWE
except ImportError:  # pragma: no cover - depends on diffusers version.
    from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder

    LTX2VocoderWithBWE = None


def _dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def _cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)


class MockLTX23TailPipeline(torch.nn.Module):
    """Minimal LTX-2.3 tail pipeline.

    The real pipeline has already finished denoise when it reaches this shape:
    packed normalized video/audio latents are unpacked/denormalized, decoded,
    video is postprocessed, and audio is decoded by audio VAE + vocoder.
    """

    def __init__(
        self,
        model: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        batch_size: int = 1,
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.batch_size = batch_size

        self.vae = AutoencoderKLLTX2Video.from_pretrained(
            model,
            subfolder="vae",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).to(device)
        self.audio_vae = AutoencoderKLLTX2Audio.from_pretrained(
            model,
            subfolder="audio_vae",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).to(device)

        vocoder_cls = LTX2VocoderWithBWE or LTX2Vocoder
        try:
            self.vocoder = vocoder_cls.from_pretrained(
                model,
                subfolder="vocoder",
                torch_dtype=dtype,
                local_files_only=local_files_only,
            ).to(device)
        except (TypeError, OSError, ValueError):
            self.vocoder = LTX2Vocoder.from_pretrained(
                model,
                subfolder="vocoder",
                torch_dtype=dtype,
                local_files_only=local_files_only,
            ).to(device)

        self.eval()

        self.vae_spatial_compression_ratio = int(self.vae.spatial_compression_ratio)
        self.vae_temporal_compression_ratio = int(self.vae.temporal_compression_ratio)
        self.audio_vae_mel_compression_ratio = int(self.audio_vae.mel_compression_ratio)
        self.audio_vae_temporal_compression_ratio = int(self.audio_vae.temporal_compression_ratio)
        self.audio_sampling_rate = int(self.audio_vae.config.sample_rate)
        self.audio_hop_length = int(self.audio_vae.config.mel_hop_length)

        self.transformer_spatial_patch_size = self._read_transformer_int("patch_size", 1)
        self.transformer_temporal_patch_size = self._read_transformer_int("patch_size_t", 1)

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)
        self.async_video_postprocessor = _LTX23VideoPostprocessor(self.video_processor)
        self.async_video_postprocessor.prealloc_shape = (
            batch_size,
            num_frames,
            height,
            width,
            int(self.vae.config.out_channels),
        )
        self.async_video_postprocessor.preallocate(device)

    def _read_transformer_int(self, key: str, default: int) -> int:
        config_path = Path(self.model) / "transformer" / "config.json"
        if not config_path.exists():
            return default
        try:
            with config_path.open("r", encoding="utf-8") as f:
                return int(json.load(f).get(key, default))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return default

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
        latents = latents.reshape(
            batch_size,
            num_frames,
            height,
            width,
            -1,
            patch_size_t,
            patch_size,
            patch_size,
        )
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7)
        return latents.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    @staticmethod
    def _denormalize_latents(
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        scaling_factor: float = 1.0,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        return latents * latents_std / scaling_factor + latents_mean

    @staticmethod
    def _unpack_audio_latents(latents: torch.Tensor, latent_length: int, num_mel_bins: int) -> torch.Tensor:
        return latents.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)

    @staticmethod
    def _denormalize_audio_latents(
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.to(latents.device, latents.dtype)
        latents_std = latents_std.to(latents.device, latents.dtype)
        return latents * latents_std + latents_mean

    def make_inputs(self, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        latent_frames = (self.num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = self.height // self.vae_spatial_compression_ratio
        latent_width = self.width // self.vae_spatial_compression_ratio
        post_patch_frames = latent_frames // self.transformer_temporal_patch_size
        post_patch_height = latent_height // self.transformer_spatial_patch_size
        post_patch_width = latent_width // self.transformer_spatial_patch_size
        video_features = (
            int(self.vae.config.latent_channels)
            * self.transformer_temporal_patch_size
            * self.transformer_spatial_patch_size
            * self.transformer_spatial_patch_size
        )
        video_shape = (
            self.batch_size,
            post_patch_frames * post_patch_height * post_patch_width,
            video_features,
        )

        duration_s = self.num_frames / float(self.fps)
        audio_latents_per_second = (
            self.audio_sampling_rate
            / self.audio_hop_length
            / float(self.audio_vae_temporal_compression_ratio)
        )
        audio_length = round(duration_s * audio_latents_per_second)
        latent_mel_bins = int(self.audio_vae.config.mel_bins) // self.audio_vae_mel_compression_ratio
        audio_shape = (
            self.batch_size,
            audio_length,
            int(self.audio_vae.config.latent_channels) * latent_mel_bins,
        )

        return {
            "video_latents": randn_tensor(
                video_shape,
                generator=generator,
                device=self.device,
                dtype=torch.float32,
            ),
            "audio_latents": randn_tensor(
                audio_shape,
                generator=generator,
                device=self.device,
                dtype=torch.float32,
            ),
        }

    def unpack_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latent_frames = (self.num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = self.height // self.vae_spatial_compression_ratio
        latent_width = self.width // self.vae_spatial_compression_ratio
        latents = self._unpack_latents(
            latents,
            latent_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )
        return self._denormalize_latents(
            latents,
            self.vae.latents_mean,
            self.vae.latents_std,
            self.vae.config.scaling_factor,
        )

    def unpack_audio_latents(self, latents: torch.Tensor) -> torch.Tensor:
        audio_length = latents.shape[1]
        latent_mel_bins = int(self.audio_vae.config.mel_bins) // self.audio_vae_mel_compression_ratio
        latents = self._denormalize_audio_latents(
            latents,
            self.audio_vae.latents_mean,
            self.audio_vae.latents_std,
        )
        return self._unpack_audio_latents(latents, audio_length, latent_mel_bins)

    def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        latents = self.unpack_video_latents(latents).to(self.vae.dtype)
        timestep_decode = None
        if getattr(self.vae.config, "timestep_conditioning", False):
            timestep_decode = torch.zeros(self.batch_size, device=self.device, dtype=latents.dtype)
        return self.vae.decode(latents, timestep_decode, return_dict=False)[0]

    def decode_audio(self, audio_latents: torch.Tensor) -> torch.Tensor:
        audio_latents = self.unpack_audio_latents(audio_latents).to(self.audio_vae.dtype)
        generated_mel_spectrograms = self.audio_vae.decode(audio_latents, return_dict=False)[0]
        return self.vocoder(generated_mel_spectrograms)

    def submit_video_postprocess_on_stream(self, video: torch.Tensor) -> tuple[Any, torch.Tensor]:
        """Enqueue optimized video postprocess on a side stream without a worker thread."""
        with record_function("ltx23.stream_video_postprocess.enqueue"):
            ready_event = current_omni_platform.Event()
            ready_event.record(current_omni_platform.current_stream(video.device))
            stream = current_omni_platform.Stream(device=video.device)
            done_event = current_omni_platform.Event()

            with torch.inference_mode(), current_omni_platform.stream(stream):
                stream.wait_event(ready_event)
                if getattr(getattr(self.video_processor, "config", None), "do_normalize", True):
                    with record_function("ltx23.video_postprocess.denormalize"):
                        video.mul_(0.5).add_(0.5).clamp_(0, 1)
                video = video.permute(0, 2, 3, 4, 1)
                with record_function("ltx23.video_postprocess.gpu_convert"):
                    device_buffer = self.async_video_postprocessor._get_device_buffer(
                        video.shape,
                        torch.float32,
                        video.device,
                    )
                    device_buffer.copy_(video)
                host_video = self.async_video_postprocessor._get_host_buffer(
                    video.shape,
                    torch.float32,
                )
                with record_function("ltx23.video_postprocess.dtoh"):
                    host_video.copy_(device_buffer, non_blocking=True)
                done_event.record(stream)

        return done_event, host_video

    @torch.inference_mode()
    def forward(
        self,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        *,
        postprocess_mode: str,
    ) -> tuple[Any, torch.Tensor]:
        with record_function("ltx23.decode"):
            video = self.decode_video(video_latents)

        video_future = None
        video_done_event = None
        video_host_buffer = None
        if video.numel() > 0:
            if postprocess_mode == "sync":
                with record_function("ltx23.video_postprocess.task"):
                    video = self.video_processor.postprocess_video(video, output_type="np")
            elif postprocess_mode == "optimized-sync":
                if not self.async_video_postprocessor.can_run(video, "np"):
                    raise RuntimeError("Optimized video postprocess is not available for this device/output_type.")
                with record_function("ltx23.video_postprocess.task"):
                    video = self.async_video_postprocessor.submit(video).result()
            elif postprocess_mode == "stream-async":
                if not self.async_video_postprocessor.can_run(video, "np"):
                    raise RuntimeError("Stream async video postprocess is not available for this device/output_type.")
                video_done_event, video_host_buffer = self.submit_video_postprocess_on_stream(video)
            elif postprocess_mode == "async":
                if not self.async_video_postprocessor.can_run(video, "np"):
                    raise RuntimeError("Async video postprocess is not available for this device/output_type.")
                video_future = self.async_video_postprocessor.submit(video)
            else:
                raise ValueError(f"Unknown postprocess_mode: {postprocess_mode}")

        with record_function("ltx23.audio_decode"):
            audio = self.decode_audio(audio_latents)

        if video_future is not None:
            with record_function("ltx23.video_postprocess.wait"):
                video = video_future.result()
        elif video_done_event is not None:
            with record_function("ltx23.video_postprocess.wait"):
                video_done_event.synchronize()
            with record_function("ltx23.video_postprocess.cpu_copy"):
                video = video_host_buffer.numpy().copy()

        return video, audio


def export_trace(
    pipeline: MockLTX23TailPipeline,
    inputs: dict[str, torch.Tensor],
    *,
    postprocess_mode: str,
    trace_path: Path,
    with_stack: bool,
    record_shapes: bool,
) -> None:
    activities = [ProfilerActivity.CPU]
    if pipeline.device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    _cuda_synchronize(pipeline.device)
    with profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=False,
        with_stack=with_stack,
    ) as prof:
        with record_function(f"mock_ltx23_tail.forward.{postprocess_mode}"):
            video, audio = pipeline(
                inputs["video_latents"],
                inputs["audio_latents"],
                postprocess_mode=postprocess_mode,
            )
        _cuda_synchronize(pipeline.device)
        del video, audio

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))
    print(f"Wrote trace {trace_path}")


def benchmark_mode(
    pipeline: MockLTX23TailPipeline,
    inputs: dict[str, torch.Tensor],
    *,
    postprocess_mode: str,
    warmup_runs: int,
    runs: int,
    empty_cache_between_runs: bool,
    trace_path: Path | None,
    trace_with_stack: bool,
    trace_record_shapes: bool,
) -> dict[str, float]:
    device = pipeline.device
    label = postprocess_mode
    latencies_ms: list[float] = []
    peak_allocated_mb: list[float] = []
    peak_reserved_mb: list[float] = []

    for i in range(warmup_runs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _cuda_synchronize(device)
        video, audio = pipeline(
            inputs["video_latents"],
            inputs["audio_latents"],
            postprocess_mode=postprocess_mode,
        )
        _cuda_synchronize(device)
        del video, audio
        if empty_cache_between_runs and device.type == "cuda":
            torch.cuda.empty_cache()

    if trace_path is not None:
        export_trace(
            pipeline,
            inputs,
            postprocess_mode=postprocess_mode,
            trace_path=trace_path,
            with_stack=trace_with_stack,
            record_shapes=trace_record_shapes,
        )

    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _cuda_synchronize(device)
        start = time.perf_counter()
        video, audio = pipeline(
            inputs["video_latents"],
            inputs["audio_latents"],
            postprocess_mode=postprocess_mode,
        )
        _cuda_synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        del video, audio

        latencies_ms.append(elapsed_ms)
        if device.type == "cuda":
            peak_allocated_mb.append(torch.cuda.max_memory_allocated(device) / 1024**2)
            peak_reserved_mb.append(torch.cuda.max_memory_reserved(device) / 1024**2)

        if empty_cache_between_runs and device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "mode": label,
        "runs": float(runs),
        "latency_ms_mean": _mean(latencies_ms),
        "latency_ms_median": _median(latencies_ms),
        "latency_ms_p95": _percentile(latencies_ms, 95),
        "latency_ms_min": min(latencies_ms) if latencies_ms else 0.0,
        "latency_ms_max": max(latencies_ms) if latencies_ms else 0.0,
        "peak_allocated_mb_mean": _mean(peak_allocated_mb),
        "peak_reserved_mb_mean": _mean(peak_reserved_mb),
    }
    print(
        f"{label:<5} latency median={result['latency_ms_median']:.2f} ms "
        f"mean={result['latency_ms_mean']:.2f} ms "
        f"p95={result['latency_ms_p95']:.2f} ms "
        f"peak_reserved={result['peak_reserved_mb_mean']:.0f} MiB"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/models/Lightricks/LTX-2.3-Diffusers")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", default="float16", choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=["sync", "optimized-sync", "stream-async", "async", "both", "all"],
        default="both",
        help=(
            "sync=diffusers VideoProcessor serial path; "
            "optimized-sync=same optimized postprocess as async but serialized before audio; "
            "stream-async=optimized postprocess enqueued on a side stream without a worker thread; "
            "async=real pipeline video postprocess helper overlapped with audio; "
            "both=sync+async; all=sync+optimized-sync+stream-async+async."
        ),
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--trace-dir",
        default="",
        help="Directory for Chrome traces. If set, exports one post-warmup trace per selected mode.",
    )
    parser.add_argument(
        "--trace-modes",
        default="",
        help="Comma-separated modes to trace. Defaults to all selected modes when --trace-dir is set.",
    )
    parser.add_argument("--trace-with-stack", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trace-record-shapes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--empty-cache-between-runs",
        action="store_true",
        help="Call torch.cuda.empty_cache() after each run. Disabled by default to match serving behavior.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault(LTX23_ASYNC_VIDEO_POSTPROCESS_ENV, "1")
    torch.set_grad_enabled(False)

    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)

    print("Loading mock tail pipeline components...")
    pipeline = MockLTX23TailPipeline(
        args.model,
        device=device,
        dtype=dtype,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        fps=args.fps,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
    )
    inputs = pipeline.make_inputs(generator)

    video_latents = inputs["video_latents"]
    audio_latents = inputs["audio_latents"]
    print(
        "Shapes: "
        f"video_latents={tuple(video_latents.shape)} "
        f"audio_latents={tuple(audio_latents.shape)} "
        f"output={args.width}x{args.height}x{args.num_frames}@{args.fps}"
    )

    results = []
    modes = {
        "sync": ["sync"],
        "optimized-sync": ["optimized-sync"],
        "stream-async": ["stream-async"],
        "async": ["async"],
        "both": ["sync", "async"],
        "all": ["sync", "optimized-sync", "stream-async", "async"],
    }[args.mode]

    for mode in modes:
        trace_path = None
        if args.trace_dir:
            trace_modes = set(args.trace_modes.split(",")) if args.trace_modes else set(modes)
            if mode in trace_modes:
                trace_path = Path(args.trace_dir) / f"ltx23_mock_tail_{mode}.json"
        results.append(
            benchmark_mode(
                pipeline,
                inputs,
                postprocess_mode=mode,
                warmup_runs=args.warmup_runs,
                runs=args.runs,
                empty_cache_between_runs=args.empty_cache_between_runs,
                trace_path=trace_path,
                trace_with_stack=args.trace_with_stack,
                trace_record_shapes=args.trace_record_shapes,
            )
        )

    summary: dict[str, Any] = {
        "model": args.model,
        "width": args.width,
        "height": args.height,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "results": results,
    }

    by_mode = {r["mode"]: r for r in results}
    if "sync" in by_mode and "async" in by_mode:
        sync_ms = by_mode["sync"]["latency_ms_median"]
        async_ms = by_mode["async"]["latency_ms_median"]
        summary["comparison"] = {
            "latency_ms_sync_median": sync_ms,
            "latency_ms_async_median": async_ms,
            "latency_ms_saved": sync_ms - async_ms,
            "latency_reduction_pct": (sync_ms - async_ms) / sync_ms * 100.0 if sync_ms > 0 else 0.0,
            "speedup": sync_ms / async_ms if async_ms > 0 else 0.0,
        }
        cmp = summary["comparison"]
        print(
            "comparison "
            f"saved={cmp['latency_ms_saved']:.2f} ms "
            f"reduction={cmp['latency_reduction_pct']:.2f}% "
            f"speedup={cmp['speedup']:.3f}x"
        )
    if {"sync", "optimized-sync", "async"}.issubset(by_mode):
        sync_ms = by_mode["sync"]["latency_ms_median"]
        optimized_sync_ms = by_mode["optimized-sync"]["latency_ms_median"]
        async_ms = by_mode["async"]["latency_ms_median"]
        total_saved = sync_ms - async_ms
        implementation_saved = sync_ms - optimized_sync_ms
        overlap_saved = optimized_sync_ms - async_ms
        summary["decomposition"] = {
            "sync_diffusers_ms": sync_ms,
            "optimized_sync_ms": optimized_sync_ms,
            "async_overlap_ms": async_ms,
            "total_saved_ms": total_saved,
            "implementation_saved_ms": implementation_saved,
            "overlap_saved_ms": overlap_saved,
            "implementation_saved_share_pct": (
                implementation_saved / total_saved * 100.0 if total_saved > 0 else 0.0
            ),
            "overlap_saved_share_pct": overlap_saved / total_saved * 100.0 if total_saved > 0 else 0.0,
        }
        dec = summary["decomposition"]
        print(
            "decomposition "
            f"implementation_saved={dec['implementation_saved_ms']:.2f} ms "
            f"({dec['implementation_saved_share_pct']:.1f}%) "
            f"overlap_saved={dec['overlap_saved_ms']:.2f} ms "
            f"({dec['overlap_saved_share_pct']:.1f}%)"
        )
    if {"optimized-sync", "stream-async", "async"}.issubset(by_mode):
        optimized_sync_ms = by_mode["optimized-sync"]["latency_ms_median"]
        stream_async_ms = by_mode["stream-async"]["latency_ms_median"]
        thread_async_ms = by_mode["async"]["latency_ms_median"]
        summary["stream_async_comparison"] = {
            "optimized_sync_ms": optimized_sync_ms,
            "stream_async_ms": stream_async_ms,
            "helper_async_ms": thread_async_ms,
            "stream_overlap_saved_ms": optimized_sync_ms - stream_async_ms,
            "helper_extra_saved_ms": stream_async_ms - thread_async_ms,
        }
        stream_cmp = summary["stream_async_comparison"]
        print(
            "stream_async "
            f"overlap_saved_no_thread={stream_cmp['stream_overlap_saved_ms']:.2f} ms "
            f"helper_extra_saved={stream_cmp['helper_extra_saved_ms']:.2f} ms"
        )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
