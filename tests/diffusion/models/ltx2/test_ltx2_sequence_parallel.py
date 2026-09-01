# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from vllm_omni.diffusion.models.ltx2.ltx2_sequence_parallel import (
    LTX2VideoToAudioParallelAttention,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _run_video_to_audio_parity(rank: int, world_size: int, master_port: int) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{master_port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(17)
        query = torch.randn(1, 3, 4, 8)
        global_key = torch.randn(1, 6, 4, 8)
        global_value = torch.randn(1, 6, 4, 8)
        key = global_key.chunk(world_size, dim=1)[rank].contiguous()
        value = global_value.chunk(world_size, dim=1)[rank].contiguous()
        sp_group = SimpleNamespace(
            ring_world_size=1,
            ulysses_group=dist.group.WORLD,
            ulysses_world_size=world_size,
            ulysses_rank=rank,
        )
        strategy = LTX2VideoToAudioParallelAttention(sp_group)

        local_query, local_key, local_value, _, ctx = strategy.pre_attention(query, key, value, None)
        local_output = F.scaled_dot_product_attention(
            local_query.transpose(1, 2),
            local_key.transpose(1, 2),
            local_value.transpose(1, 2),
        ).transpose(1, 2)
        actual = strategy.post_attention(local_output, ctx)
        expected = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            global_key.transpose(1, 2),
            global_value.transpose(1, 2),
        ).transpose(1, 2)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    finally:
        dist.destroy_process_group()


def test_ltx_video_to_audio_sp_matches_replicated_attention(unused_tcp_port):
    torch.multiprocessing.spawn(
        _run_video_to_audio_parity,
        args=(2, unused_tcp_port),
        nprocs=2,
    )


def test_ltx_video_to_audio_sp_rejects_ring_parallelism():
    sp_group = SimpleNamespace(ring_world_size=2)

    with pytest.raises(NotImplementedError, match="ring_degree must be 1"):
        LTX2VideoToAudioParallelAttention(sp_group)
