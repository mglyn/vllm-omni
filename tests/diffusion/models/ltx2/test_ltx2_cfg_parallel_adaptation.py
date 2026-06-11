from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_pipeline(sequence_parallel_size: int = 1) -> LTX2Pipeline:
    pipeline = object.__new__(LTX2Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.audio_vae_temporal_compression_ratio = 4
    pipeline.audio_vae_mel_compression_ratio = 4
    pipeline.od_config = SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=sequence_parallel_size))
    # Mock audio_vae with identity normalization (mean=0, std=1) so
    # _normalize_audio_latents is a no-op and test values are preserved.
    pipeline.audio_vae = SimpleNamespace(
        latents_mean=torch.tensor(0.0),
        latents_std=torch.tensor(1.0),
    )
    return pipeline


def test_prepare_audio_latents_pads_packed_sequence_dim_for_provided_latents():
    pipeline = _make_pipeline(sequence_parallel_size=4)
    latents = torch.arange(40, dtype=torch.float32).view(1, 10, 4)

    padded, original_num_frames, padded_num_frames = pipeline.prepare_audio_latents(
        batch_size=1,
        num_channels_latents=2,
        num_mel_bins=8,
        audio_latent_length=10,
        dtype=torch.float32,
        device=torch.device("cpu"),
        latents=latents,
    )

    assert original_num_frames == 10
    assert padded_num_frames == 12
    assert padded.shape == (1, 12, 4)
    torch.testing.assert_close(padded[:, :10], latents)
    torch.testing.assert_close(padded[:, 10:], torch.zeros(1, 2, 4))


def test_unpad_audio_latents_restores_original_frames_before_unpack():
    pipeline = _make_pipeline()
    original = torch.arange(40, dtype=torch.float32).view(1, 10, 4)
    padded = torch.cat([original, torch.full((1, 2, 4), 999.0)], dim=1)

    unpadded = pipeline._unpad_audio_latents(padded, 10)
    unpacked = pipeline._unpack_audio_latents(unpadded, latent_length=10, num_mel_bins=2)
    expected = pipeline._unpack_audio_latents(original, latent_length=10, num_mel_bins=2)

    assert unpacked.shape == (1, 2, 10, 2)
    assert not (unpacked == 999.0).any()
    torch.testing.assert_close(unpacked, expected)


def test_ltx2_pipeline_loads_distributed_video_vae(monkeypatch, tmp_path):
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import (
        DistributedAutoencoderKLLTX2Video,
    )
    from vllm_omni.diffusion.models.ltx2 import pipeline_ltx2 as ltx2

    loaded_subfolders = {}

    class FakeModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()

        def to(self, *_args, **_kwargs):
            return self

    class FakeVideoVae(FakeModule):
        spatial_compression_ratio = 32
        temporal_compression_ratio = 8

    class FakeAudioVae(FakeModule):
        mel_compression_ratio = 4
        temporal_compression_ratio = 4

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(sample_rate=16000, mel_hop_length=160)

    class FakeTransformer(FakeModule):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(patch_size=1, patch_size_t=1)

    def fake_from_pretrained_with_prefetch(from_pretrained, *_args, subfolder, **_kwargs):
        loaded_subfolders[subfolder] = from_pretrained
        if subfolder == "vae":
            return FakeVideoVae()
        if subfolder == "audio_vae":
            return FakeAudioVae()
        return FakeModule()

    monkeypatch.setattr(ltx2, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(ltx2, "prefetch_subfolders", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ltx2, "from_pretrained_with_prefetch", fake_from_pretrained_with_prefetch)
    monkeypatch.setattr(
        ltx2.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(model_max_length=1024),
    )
    monkeypatch.setattr(
        ltx2.FlowMatchEulerDiscreteScheduler,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(ltx2, "load_transformer_config", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(ltx2, "create_transformer_from_config", lambda *_args, **_kwargs: FakeTransformer())

    od_config = SimpleNamespace(model=str(tmp_path), dtype=torch.float32, quantization_config=None)
    pipe = LTX2Pipeline(od_config=od_config)

    assert getattr(loaded_subfolders["vae"], "__self__", None) is DistributedAutoencoderKLLTX2Video
    assert isinstance(pipe.vae, FakeVideoVae)
