# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from dataclasses import asdict, replace

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.layers.gemm_tuning.backends import PadRows, Tile
from vllm_omni.diffusion.layers.gemm_tuning.profile import Target, load_configs, profile_entries

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def table(target):
    return {
        "version": 1,
        "operation": "bf16_nt_no_bias",
        "target": asdict(target),
        "entries": {
            "37,128,64": {"backend": "triton", "config": asdict(Tile(32, 64, 32, 4, 3))},
            "37,64,128": {"backend": "pad_rows", "config": {"multiple": 16}},
            "36,64,128": {"backend": "default"},
        },
    }


@pytest.mark.cpu
def test_target_and_exact_shape(tmp_path):
    target = Target("cuda", "GPU", "80", "torch", "triton", "cuda", 108, "vllm")
    data = table(target)
    (tmp_path / target.filename).write_text(json.dumps(data))
    assert load_configs(tmp_path, target) == {
        (37, 128, 64): Tile(32, 64, 32, 4, 3),
        (37, 64, 128): PadRows(16),
    }
    for other in [replace(target, arch="90"), replace(target, compute_units=14), replace(target, vllm_version="next")]:
        assert load_configs(tmp_path, other) == {}
        with pytest.raises(ValueError):
            profile_entries(data, other)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "key,row",
    [
        ("037,64,64", {"backend": "default"}),
        ("0,64,64", {"backend": "default"}),
        ("37,64,64", {"backend": "custom"}),
        ("37,64,64", {"backend": "pad_rows", "config": {"multiple": -1}}),
        ("37,64,64", {"backend": "triton", "config": asdict(Tile(32, 64, 32, 4, 3)) | {"num_warps": 5}}),
    ],
)
def test_reject_invalid_entry(key, row):
    target = Target("cuda", "GPU", "80", "torch", "triton", "cuda", 108, "vllm")
    with pytest.raises(ValueError):
        profile_entries(table(target) | {"entries": {key: row}}, target)


@hardware_test(res={"cuda": ["H100", "B200"]})
@torch.inference_mode()
def test_bagel_mlp_dispatch(tmp_path, monkeypatch):
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.utils.network_utils import get_file_store_init_method

    from vllm_omni.diffusion.layers.gemm_tuning import runtime
    from vllm_omni.diffusion.models.bagel.bagel_transformer import BagelMLP

    device = torch.device("cuda")
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0, distributed_init_method=get_file_store_init_method()
    )
    try:
        initialize_model_parallel()
        model = BagelMLP(hidden_size=64, intermediate_size=128).to(device=device, dtype=torch.bfloat16)
        for p in model.parameters():
            p.copy_(torch.randint(-1, 2, p.shape, device=device))
        x = torch.randint(-1, 2, (1, 37, 64), device=device).to(torch.bfloat16)
        baseline = model(x)
        target = Target.current(device)
        data = table(target)
        data["entries"]["37,256,64"] = data["entries"].pop("37,128,64")
        (tmp_path / target.filename).write_text(json.dumps(data))
        replay = BagelMLP(hidden_size=64, intermediate_size=128).to(device=device, dtype=torch.bfloat16)
        replay.load_state_dict(model.state_dict())
        calls = []

        def record_run(run):
            def record(config, x, weight):
                calls.append(config)
                return run(config, x, weight)

            return record

        for backend in (Tile, PadRows):
            monkeypatch.setattr(backend, "run", record_run(backend.run))
        runtime.install(replay, device, {"mode": "dispatch", "directory": str(tmp_path)})
        assert torch.equal(replay(x), baseline)
        assert calls == [Tile(32, 64, 32, 4, 3), PadRows(16)]
        calls.clear()
        assert torch.equal(replay(x[:, :36]), model(x[:, :36]))
        assert calls == []
        proj = replay.gate_up_proj
        bias = x.new_ones(proj.weight.size(0))
        strided = x.transpose(-2, -1).contiguous().transpose(-2, -1)
        for activation, extra_bias in [(x, bias), (strided, None)]:
            got = proj.quant_method.apply(proj, activation, extra_bias)
            assert torch.equal(got, torch.nn.functional.linear(activation, proj.weight, extra_bias))
        assert calls == []
    finally:
        cleanup_dist_env_and_memory()
