# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.triton_utils import HAS_TRITON

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.diffusion]


def _sm90_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _qk_reference(x, cos, sin):
    normalized = F.rms_norm(x.float(), (64,), None, 1e-5).to(x.dtype)
    rotary = normalized[..., :48]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat(
        (rotary * cos + rotated * sin, normalized[..., 48:]),
        dim=-1,
    )


@pytest.mark.skipif(not _sm90_available(), reason="CUDA SM90 required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize(("batch", "sequence"), [(1, 1), (1, 195), (2, 1797)])
def test_h3_vae_qk_norm_rope_is_bit_exact(batch, sequence):
    from vllm_omni.diffusion.models.minimax_h3.ops.vae.qk_norm_rope import (
        try_qk_norm_rope_exact,
    )

    torch.manual_seed(17)
    qkv = torch.randn(
        batch,
        sequence,
        32,
        192,
        device="cuda",
        dtype=torch.float16,
    )
    q, k, _ = qkv.chunk(3, dim=-1)
    cos = torch.randn(batch, sequence, 1, 48, device="cuda", dtype=torch.float16)
    sin = torch.randn_like(cos)

    expected_q = _qk_reference(q, cos, sin)
    expected_k = _qk_reference(k, cos, sin)
    with torch.inference_mode():
        actual = try_qk_norm_rope_exact(q, k, (cos, sin), 1e-5)

    assert actual is not None
    actual_q, actual_k = actual
    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_k, expected_k)


@pytest.mark.skipif(not _sm90_available(), reason="CUDA SM90 required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_h3_vae_scaled_residual_is_bit_exact():
    from vllm_omni.diffusion.models.minimax_h3.ops.vae.scaled_residual import (
        try_scaled_residual_exact,
    )

    torch.manual_seed(29)
    residual = torch.randn(195, 2048, device="cuda", dtype=torch.float32)
    branch = torch.randn(195, 2048, device="cuda", dtype=torch.float16)
    scale = torch.randn(2048, device="cuda", dtype=torch.float32)
    expected = residual + branch * scale

    with torch.inference_mode():
        actual = try_scaled_residual_exact(residual, branch, scale)

    assert actual is not None
    assert torch.equal(actual, expected)


def test_h3_vae_exact_ops_reject_unsupported_inputs():
    from vllm_omni.diffusion.models.minimax_h3.ops.vae.qk_norm_rope import (
        try_qk_norm_rope_exact,
    )
    from vllm_omni.diffusion.models.minimax_h3.ops.vae.scaled_residual import try_scaled_residual_exact

    q = torch.randn(1, 2, 32, 64, dtype=torch.float32)
    cos = torch.randn(1, 2, 1, 48, dtype=torch.float32)
    residual = torch.randn(2, 2048)
    branch = torch.randn(2, 2048, dtype=torch.float16)
    scale = torch.randn(2048)
    with torch.inference_mode():
        assert try_qk_norm_rope_exact(q, q, (cos, cos), 1e-5) is None
        assert try_scaled_residual_exact(residual, branch, scale) is None


def _make_decoder():
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_qkv = nn.Linear(8, 24)
            self.to_out = nn.Linear(8, 8)
            self.norm_q = nn.RMSNorm(8, elementwise_affine=False)
            self.norm_k = nn.RMSNorm(8, elementwise_affine=False)
            self.spatial_parallel = False
            self.dim_head = 8

        def perform_attention(self, query, _key, _value, _pack_info):
            return query

        def forward(self, hidden_states, rotary_pos_emb=None, pack_info=None):
            return hidden_states

    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(8, 32)
            self.w2 = nn.Linear(16, 8)
            self.use_gated = True
            self.act_fn = nn.SiLU()

        def forward(self, hidden_states):
            hidden_states = self.w1(hidden_states)
            gate, hidden_states = hidden_states.chunk(2, dim=-1)
            return self.w2(self.act_fn(gate) * hidden_states)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention()
            self.ff = FeedForward()
            self.norm1 = nn.RMSNorm(8)
            self.norm2 = nn.RMSNorm(8)
            self.scale1 = nn.Parameter(torch.zeros(8))
            self.scale2 = nn.Parameter(torch.zeros(8))
            self.use_scale = True

        def forward(self, hidden_states, rotary_pos_emb=None, pack_info=None):
            return hidden_states

    decoder = nn.Module()
    decoder.transformer_blocks = nn.ModuleList([Block(), Block()])
    decoder.proj_out = nn.Linear(8, 8)
    return decoder


def test_h3_vae_install_precasts_only_block_linears(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.ops import vae as vae_ops

    monkeypatch.setattr(vae_ops, "_supports_h3_vae_optimizations", lambda _device: True)
    decoder = _make_decoder()

    assert vae_ops.install_h3_vae_optimizations(
        decoder,
        device=torch.device("cuda:0"),
    )

    for block in decoder.transformer_blocks:
        assert block.attn.to_qkv.weight.dtype == torch.float16
        assert block.attn.to_out.weight.dtype == torch.float16
        assert block.ff.w1.weight.dtype == torch.float16
        assert block.ff.w2.weight.dtype == torch.float16
        assert block.forward.__func__.__name__ == "_optimized_transformer_block"
        assert block.attn.forward.__func__.__name__ == "_optimized_attention"
        assert block.ff.forward.__func__.__name__ == "_optimized_feed_forward"
    assert decoder.proj_out.weight.dtype == torch.float32

    # Repeated installation is idempotent.
    assert vae_ops.install_h3_vae_optimizations(
        decoder,
        device=torch.device("cuda:0"),
    )


@pytest.mark.skipif(not _sm90_available(), reason="CUDA SM90 required")
def test_h3_vae_swiglu_uses_post_linear_fp16_output(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.ops import vae as vae_ops

    monkeypatch.setattr(vae_ops, "_supports_h3_vae_optimizations", lambda _device: True)
    decoder = _make_decoder().cuda()
    feed_forward = decoder.transformer_blocks[0].ff
    reference_forward = type(feed_forward).forward
    assert vae_ops.install_h3_vae_optimizations(
        decoder,
        device=torch.device("cuda:0"),
    )

    hidden_states = torch.randn(4, 8, device="cuda", dtype=torch.float32)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        expected = reference_forward(feed_forward, hidden_states)
        actual = feed_forward(hidden_states)

    assert expected.dtype == torch.float16
    assert torch.equal(actual, expected)


def test_h3_vae_install_leaves_unsupported_target_untouched(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.ops import vae as vae_ops

    monkeypatch.setattr(vae_ops, "_supports_h3_vae_optimizations", lambda _device: False)
    decoder = _make_decoder()

    assert not vae_ops.install_h3_vae_optimizations(
        decoder,
        device=torch.device("cuda:0"),
    )
    assert not hasattr(decoder, "_omni_h3_vae_optimizations_installed")
    for block in decoder.transformer_blocks:
        assert block.attn.to_qkv.weight.dtype == torch.float32
        assert block.ff.w1.weight.dtype == torch.float32
        assert block.forward.__func__.__name__ == "forward"
        assert block.attn.forward.__func__.__name__ == "forward"


def test_h3_vae_support_is_platform_and_capability_scoped(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.ops import vae as vae_ops

    class FakePlatform:
        def __init__(self, cuda, capability):
            self.cuda = cuda
            self.capability = capability

        def is_cuda(self):
            return self.cuda

        def is_available(self):
            return True

        def get_device_capability(self, _device_index):
            capability = self.capability

            class Capability:
                def to_int(self):
                    return capability

            return Capability()

    platform = FakePlatform(True, 90)
    monkeypatch.setattr(vae_ops, "current_omni_platform", platform)
    monkeypatch.setattr(vae_ops, "HAS_TRITON", True)

    assert vae_ops._supports_h3_vae_optimizations(torch.device("cuda:0"))
    platform.capability = 89
    assert not vae_ops._supports_h3_vae_optimizations(torch.device("cuda:0"))
    platform.cuda = False
    assert not vae_ops._supports_h3_vae_optimizations(torch.device("cuda:0"))
    assert not vae_ops._supports_h3_vae_optimizations(torch.device("cpu"))
