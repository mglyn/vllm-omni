# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.ltx2.ltx2_components import (
    LTX25_DFR_COMPONENT_PROFILE,
    resolve_ltx_checkpoint_kind,
    resolve_ltx_component_profile,
)
from vllm_omni.diffusion.models.ltx2.ltx2_denoise import LTXDenoiseContext
from vllm_omni.diffusion.models.ltx2.ltx2_recipes import (
    LTX25_DFR_RECIPE,
    LTX_DETAILING_ADAPTER_SLOT,
    resolve_ltx_pipeline_recipe,
)
from vllm_omni.diffusion.models.ltx2.ltx2_transformer import LTX2AudioVideoRotaryPosEmbed
from vllm_omni.diffusion.models.ltx2.pipeline_ltx25_dfr import (
    LTX25DFRPipeline,
    _carry_decode_generators,
    _DFRVideoConditioning,
    resolve_dfr_canvas,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("frames", "expected_canvas", "positions"),
    [
        (121, 121, (24, 48, 72, 96, 120)),
        (97, 97, (32, 64, 96)),
        (105, 121, (24, 48, 72, 96, 120)),
    ],
)
def test_dfr_canvas_matches_official_segment_selection(frames, expected_canvas, positions):
    assert resolve_dfr_canvas(frames) == (expected_canvas, positions)


def test_dfr_recipe_is_distilled_three_stage_4k():
    assert resolve_ltx_checkpoint_kind("dfr") == "distilled"
    assert resolve_ltx_component_profile("dfr", "2.5") is LTX25_DFR_COMPONENT_PROFILE
    assert resolve_ltx_pipeline_recipe("dfr", "2.5") is LTX25_DFR_RECIPE
    assert (LTX25_DFR_RECIPE.width, LTX25_DFR_RECIPE.height) == (3840, 2176)
    stage1, stage2, epilogue = LTX25_DFR_RECIPE.phases
    assert [stage.spatial_downscale for stage in (stage1, stage2, epilogue)] == [4, 2, 1]
    assert stage1.sampler == "euler_ancestral"
    assert stage2.adapter_slot == epilogue.adapter_slot == LTX_DETAILING_ADAPTER_SLOT
    assert stage2.adapter_scale == epilogue.adapter_scale == 0.5
    assert not stage2.freeze_audio
    assert epilogue.freeze_audio
    assert LTX25_DFR_RECIPE.audio_output_phase == 0


def test_dfr_carry_decode_uses_independent_official_seed_offsets():
    request_generator = torch.Generator().manual_seed(17)
    state = request_generator.get_state()

    plane_0 = _carry_decode_generators(request_generator, 0)
    plane_3 = _carry_decode_generators(request_generator, 3)

    assert isinstance(plane_0, torch.Generator)
    assert isinstance(plane_3, torch.Generator)
    assert plane_0.initial_seed() == 4017
    assert plane_3.initial_seed() == 4020
    assert torch.equal(request_generator.get_state(), state)


def _tiny_dfr_pipe() -> LTX25DFRPipeline:
    pipe = object.__new__(LTX25DFRPipeline)
    torch.nn.Module.__init__(pipe)
    pipe.device = torch.device("cpu")
    pipe.vae_spatial_compression_ratio = 32
    pipe.vae_temporal_compression_ratio = 8
    pipe.transformer_spatial_patch_size = 1
    pipe.transformer_temporal_patch_size = 1
    pipe.transformer = SimpleNamespace(
        config=SimpleNamespace(in_channels=2),
        rope=LTX2AudioVideoRotaryPosEmbed(
            dim=24,
            scale_factors=(8, 32, 32),
            rope_type="interleaved",
        ),
    )
    pipe.vae = SimpleNamespace(
        latents_mean=torch.zeros(2),
        latents_std=torch.ones(2),
        config=SimpleNamespace(scaling_factor=1.0),
    )
    return pipe


def _inputs(*, height=32, width=32, latents=None):
    return SimpleNamespace(
        height=height,
        width=width,
        num_frames=17,
        frame_rate=24.0,
        num_videos_per_prompt=1,
        latents=latents,
        generator=torch.Generator().manual_seed(4),
        image_crf=None,
    )


def test_dfr_generated_slots_have_single_frame_coords_and_extractable_layout():
    pipe = _tiny_dfr_pipe()
    pipe._dfr_conditioning = _DFRVideoConditioning(generated_positions=(8, 16))
    prompt = SimpleNamespace(batch_size=1, positive_connector_prompt_embeds=torch.zeros(1, dtype=torch.float32))

    prepared = pipe._prepare_video_latents_stage(
        _inputs(),
        prompt,
        device=torch.device("cpu"),
        noise_scale=1.0,
    )

    assert prepared.base_video_token_count == 3
    assert prepared.latents.shape == (1, 5, 2)
    assert prepared.generated_keyframe_layout.first_token == 3
    assert prepared.generated_keyframe_layout.tokens_per_keyframe == 1
    torch.testing.assert_close(prepared.keyframes_mask[0, :, 0], torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0]))
    torch.testing.assert_close(
        prepared.video_coords[0, 0, 3:],
        torch.tensor([[8 / 24, 9 / 24], [16 / 24, 17 / 24]]),
    )


def test_dfr_reference_latent_is_clean_conditioning_and_spatially_scaled():
    pipe = _tiny_dfr_pipe()
    reference = torch.ones(1, 2, 3, 1, 1)
    pipe._dfr_conditioning = _DFRVideoConditioning(
        generated_positions=(8,),
        reference_latent=reference,
        reference_downscale_factor=2,
    )
    prompt = SimpleNamespace(batch_size=1, positive_connector_prompt_embeds=torch.zeros(1, dtype=torch.float32))
    base = torch.zeros(1, 2, 3, 2, 2)

    prepared = pipe._prepare_video_latents_stage(
        _inputs(height=64, width=64, latents=base),
        prompt,
        device=torch.device("cpu"),
        noise_scale=0.5,
    )

    # base=12, slot=4, reference=3
    assert prepared.latents.shape[1] == 19
    torch.testing.assert_close(prepared.conditioning_mask[:, :16], torch.zeros(1, 16))
    torch.testing.assert_close(prepared.conditioning_mask[:, 16:], torch.ones(1, 3))
    assert prepared.video_coords[:, 1:, 16:].amax().item() == 64


def test_dfr_epilogue_tiles_blend_every_base_token_to_one():
    pipe = _tiny_dfr_pipe()
    frames, height, width = 2, 68, 120
    base_count = frames * height * width
    coords = pipe.transformer.rope.prepare_video_coords(1, frames, height, width, torch.device("cpu"), fps=24)
    denoise_ctx = LTXDenoiseContext(
        latents=torch.zeros(1, base_count, 2),
        audio_latents=torch.zeros(1, 1, 2),
        video_coords=coords,
        audio_coords=torch.zeros(1, 1, 1, 2),
        base_video_token_count=base_count,
    )
    forward_ctx = SimpleNamespace(latent_num_frames=frames, latent_height=height, latent_width=width)

    accumulated = torch.zeros(base_count)
    tiles = pipe._spatial_tile_indices(denoise_ctx, forward_ctx)
    assert len(tiles) == 4
    for indices, blend in tiles:
        accumulated[indices[: blend.numel()]] += blend
    torch.testing.assert_close(accumulated, torch.ones_like(accumulated))
