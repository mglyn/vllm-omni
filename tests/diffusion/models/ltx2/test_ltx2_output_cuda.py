# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CUDA coverage for LTX decoded-video output preparation."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers.video_processor import VideoProcessor

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.ltx2.ltx2_runtime import _prepare_ltx2_video_output
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("do_normalize", [True, False])
def test_ltx_uint8_output_runs_on_cuda_with_bounded_rounding_drift(dtype, do_normalize):
    processor = VideoProcessor(vae_scale_factor=8, do_normalize=do_normalize)
    source = torch.linspace(
        -1.2 if do_normalize else -0.2,
        1.2,
        2 * 3 * 4 * 5 * 7,
        device="cuda",
        dtype=torch.float32,
    ).reshape(2, 3, 4, 5, 7)
    source = source.to(dtype)

    legacy = processor.postprocess_video(source.clone(), output_type="np")
    expected = np.clip(legacy, 0.0, 1.0)
    expected *= 255.0
    np.rint(expected, out=expected)
    expected = expected.astype(np.uint8)

    prepared = _prepare_ltx2_video_output(source.clone(), do_normalize=do_normalize)

    assert prepared.is_cuda
    assert prepared.shape == (2, 4, 5, 7, 3)
    assert prepared.dtype == torch.uint8
    assert prepared.is_contiguous()
    delta = np.abs(prepared.cpu().numpy().astype(np.int16) - expected.astype(np.int16))
    assert delta.max() <= 1


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("use_diffusion_decoder", [False, True], ids=["conv-vae", "diff-vae"])
def test_ltx_bf16_decoder_output_uses_cuda_transport_path(use_diffusion_decoder):
    decoded_video = torch.linspace(
        -1.2,
        1.2,
        1 * 3 * 2 * 3 * 4,
        device="cuda",
        dtype=torch.float32,
    ).reshape(1, 3, 2, 3, 4)
    decoded_video = decoded_video.to(torch.bfloat16)
    processor = VideoProcessor(vae_scale_factor=8)
    legacy = processor.postprocess_video(decoded_video.clone(), output_type="np")
    expected = np.rint(np.clip(legacy, 0.0, 1.0) * 255.0).astype(np.uint8)
    decode_calls = []

    class VideoDecoder:
        dtype = torch.bfloat16
        config = SimpleNamespace(timestep_conditioning=False)

        def __init__(self, name):
            self.name = name

        def decode(self, *_args, **_kwargs):
            decode_calls.append(self.name)
            return (decoded_video.clone(),)

    class AudioVae:
        dtype = torch.bfloat16

        def decode(self, audio_latents, *, return_dict):
            assert return_dict is False
            return (audio_latents,)

    pipe = object.__new__(LTX2Pipeline)
    torch.nn.Module.__init__(pipe)
    pipe.reports_stage_durations = False
    pipe.distributed_video_decode = False
    pipe.use_diffusion_decoder = use_diffusion_decoder
    pipe.vae = VideoDecoder("conv-vae")
    pipe.diffusion_decoder = VideoDecoder("diff-vae")
    pipe.audio_vae = AudioVae()
    pipe.vocoder = torch.nn.Identity()
    pipe.video_processor = processor

    output = pipe._decode_output(
        latents=torch.ones(1, 1, device="cuda", dtype=torch.bfloat16),
        audio_latents=torch.ones(1, 1, device="cuda", dtype=torch.bfloat16),
        output_type="np",
        connector_prompt_embeds=torch.ones(1, 1, device="cuda", dtype=torch.bfloat16),
        generator=None,
        device=torch.device("cuda"),
        decode_timestep=0.0,
        decode_noise_scale=None,
        prompt_batch_size=1,
    )

    prepared = output.output[0]
    selected_decoder = "diff-vae" if use_diffusion_decoder else "conv-vae"
    assert decode_calls == [selected_decoder]
    assert prepared.is_cuda
    assert prepared.shape == (1, 2, 3, 4, 3)
    assert prepared.dtype == torch.uint8
    assert prepared.is_contiguous()
    delta = np.abs(prepared.cpu().numpy().astype(np.int16) - expected.astype(np.int16))
    assert delta.max() <= 1
