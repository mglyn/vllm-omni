# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Shared LTX text-encoder offload contract and real Gemma stack coverage."""

import pytest
import torch

from tests.diffusion.offloader.helpers import patch_offload_runtime
from vllm_omni.diffusion.models.ltx2.ltx2_runtime import LTXRuntime
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import (
    LTX2DistilledOneStagePipeline,
    LTX2I2VDMD2Pipeline,
    LTX2Pipeline,
    LTX2T2VDMD2Pipeline,
)
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_two_stage import (
    LTX2DistilledTwoStagePipeline,
    LTX2TwoStagePipeline,
)
from vllm_omni.diffusion.models.ltx2.pipeline_ltx25_dfr import LTX25DFRPipeline
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.component_utils import get_encoder_block_groups
from vllm_omni.diffusion.offloader.distributed_layerwise_backend import DistributedLayerwiseOffloadBackend
from vllm_omni.diffusion.offloader.layerwise_backend import LayerWiseOffloadBackend
from vllm_omni.diffusion.offloader.offload_plan import get_offload_plan
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "pipeline_cls",
    [
        LTX2Pipeline,
        LTX2DistilledOneStagePipeline,
        LTX2TwoStagePipeline,
        LTX2DistilledTwoStagePipeline,
        LTX2T2VDMD2Pipeline,
        LTX2I2VDMD2Pipeline,
        LTX25DFRPipeline,
    ],
    ids=lambda cls: cls.__name__,
)
def test_ltx_entries_inherit_shared_encoder_offload_plan(pipeline_cls):
    assert get_offload_plan(pipeline_cls) is LTXRuntime._offload_plan


@pytest.mark.parametrize("family", ["Gemma3", "Gemma4Unified"])
@pytest.mark.parametrize("distributed", [False, True], ids=["layerwise", "dlo"])
def test_ltx_encoder_offload_preserves_gemma_forward(monkeypatch, family, distributed):
    import transformers

    if not hasattr(transformers, family + "ForConditionalGeneration"):
        pytest.skip(f"Installed Transformers does not provide {family}")
    text_config = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
    )
    options = {}
    if family == "Gemma3":
        options = dict(
            vision_config=dict(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                image_size=8,
                patch_size=4,
            ),
            mm_tokens_per_image=4,
        )
    else:
        text_config.update(global_head_dim=16, num_global_key_value_heads=2)
    config = getattr(transformers, family + "Config")(text_config=text_config, **options)
    encoder = getattr(transformers, family + "ForConditionalGeneration")(config).eval()
    pipeline = torch.nn.Module()
    pipeline.text_encoder = encoder
    pipeline._offload_plan = LTXRuntime._offload_plan
    groups = get_encoder_block_groups(encoder, "text_encoder", get_offload_plan(pipeline), strict=True)
    assert groups == [encoder.model.language_model.layers]
    inputs = dict(input_ids=torch.tensor([[2, 5, 7, 9]]), output_hidden_states=True, use_cache=False)

    patch_offload_runtime(monkeypatch, current_omni_platform, synchronize=True)
    backend_cls = DistributedLayerwiseOffloadBackend if distributed else LayerWiseOffloadBackend
    backend = backend_cls(
        OffloadConfig(
            strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE if distributed else OffloadStrategy.LAYER_WISE,
            components=frozenset({"text_encoder"}),
            dlo_transfers={"dit": "rank-local", "text_encoder": "allgather" if distributed else "rank-local"},
            pin_cpu_memory=False,
        ),
        torch.device("cpu"),
    )
    with torch.inference_mode():
        expected = encoder(**inputs).hidden_states[-1]
        backend.enable(pipeline)
        try:
            assert encoder._omni_layerwise_enabled
            assert len(encoder._omni_layerwise_hooks) == 3
            for _ in range(2):
                torch.testing.assert_close(encoder(**inputs).hidden_states[-1], expected, atol=0, rtol=0)
        finally:
            backend.disable()
        torch.testing.assert_close(encoder(**inputs).hidden_states[-1], expected, atol=0, rtol=0)
