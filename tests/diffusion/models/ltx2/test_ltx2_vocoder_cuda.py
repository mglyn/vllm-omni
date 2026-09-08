# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression tests for scoped LTX vocoder determinism and precision."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _cudnn_settings():
    return (
        torch.backends.cudnn.enabled,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.benchmark_limit,
        torch.backends.cudnn.allow_tf32,
    )


class _BWEConvVocoder(torch.nn.Module):
    def __init__(self, bwe, fail):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            4,
            4,
            kernel_size=3,
            bias=False,
            device="cuda",
            dtype=torch.bfloat16,
        )
        if bwe:
            self.bwe_generator = torch.nn.Identity()
        self.fail = fail
        self.input_dtype = None
        self.conv_output_dtype = None
        self.cudnn_deterministic = None

    def forward(self, value):
        self.cudnn_deterministic = torch.backends.cudnn.deterministic
        self.cudnn_settings = _cudnn_settings()
        self.input_dtype = value.dtype
        if self.fail:
            raise RuntimeError("injected vocoder failure")
        output = self.conv(value)
        self.conv_output_dtype = output.dtype
        return output


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("bwe", [False, True])
@pytest.mark.parametrize("initial_deterministic", [False, True])
@pytest.mark.parametrize("fail", [False, True])
@torch.inference_mode()
def test_ltx_vocoder_determinism_and_precision(monkeypatch, bwe, initial_deterministic, fail):
    from vllm_omni.diffusion.models.ltx2.ltx2_runtime import _run_ltx_vocoder

    vocoder = _BWEConvVocoder(bwe, fail)
    generated_mel = torch.randn((1, 4, 128), device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", initial_deterministic)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark_limit", 17)
    settings = _cudnn_settings()
    if fail:
        with pytest.raises(RuntimeError, match="injected vocoder failure"):
            _run_ltx_vocoder(vocoder, generated_mel)
    else:
        outputs = [_run_ltx_vocoder(vocoder, generated_mel) for _ in range(3)]
        assert all(torch.equal(outputs[0], output) for output in outputs[1:])
        assert outputs[0].dtype == torch.bfloat16
        assert vocoder.conv_output_dtype == (torch.float32 if bwe else torch.bfloat16)

    assert torch.backends.cudnn.deterministic == initial_deterministic
    assert _cudnn_settings() == settings
    assert vocoder.cudnn_settings == settings
    assert next(vocoder.parameters()).dtype == torch.bfloat16
    assert vocoder.cudnn_deterministic
    assert vocoder.input_dtype == (torch.float32 if bwe else torch.bfloat16)


@pytest.mark.cpu
@pytest.mark.parametrize("initial_deterministic", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_official_vocoder_determinism_restores_settings(monkeypatch, initial_deterministic, fail):
    from tests.e2e.accuracy.ltx.run_ltx25_reference import _configure_official_vocoder_determinism

    monkeypatch.setattr(torch.backends.cudnn, "deterministic", initial_deterministic)
    if torch.backends.cudnn.is_available():
        monkeypatch.setattr(torch.backends.cudnn, "benchmark_limit", 17)
    settings = _cudnn_settings()
    calls = []

    class FakeVocoder:
        def forward(self, mel):
            calls.append(mel)
            assert torch.backends.cudnn.deterministic
            assert _cudnn_settings() == settings
            if fail:
                raise RuntimeError("injected official failure")
            return mel

    module = ModuleType("ltx_core.model.audio_vae.vocoder")
    module.VocoderWithBWE = FakeVocoder
    monkeypatch.setitem(sys.modules, module.__name__, module)
    _configure_official_vocoder_determinism()
    wrapped = FakeVocoder.forward
    _configure_official_vocoder_determinism()
    assert FakeVocoder.forward is wrapped
    mel = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    if fail:
        with pytest.raises(RuntimeError, match="injected official failure"):
            FakeVocoder().forward(mel)
    else:
        assert FakeVocoder().forward(mel) is mel
    assert calls == [mel]
    assert torch.backends.cudnn.deterministic == initial_deterministic
    assert _cudnn_settings() == settings
