# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CUDA coverage for LTX decoded-video output preparation."""

import numpy as np
import pytest
import torch
from diffusers.video_processor import VideoProcessor

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.ltx2.ltx2_runtime import _prepare_ltx2_video_output

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
