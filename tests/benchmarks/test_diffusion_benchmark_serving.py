# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_diffusion_benchmark_serving():
    repo_root = Path(__file__).resolve().parents[2]
    bench_dir = repo_root / "benchmarks" / "diffusion"
    sys.path.insert(0, str(bench_dir))
    spec = importlib.util.spec_from_file_location(
        "diffusion_benchmark_serving_for_test",
        bench_dir / "diffusion_benchmark_serving.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_t2v_warmup_num_frames_can_match_measured_shape():
    module = _load_diffusion_benchmark_serving()

    req = module.RequestFuncInput(
        prompt="prompt",
        api_url="http://127.0.0.1:8098/v1/videos",
        model="default",
        width=512,
        height=384,
        num_frames=25,
        num_inference_steps=20,
    )
    args = SimpleNamespace(
        task="t2v",
        warmup_num_inference_steps=20,
        warmup_num_frames=25,
    )

    warm_req = module._make_warmup_request([req], 0, args)

    assert warm_req.num_frames == 25
    assert warm_req.num_inference_steps == 20


def test_t2v_warmup_num_frames_defaults_to_single_frame():
    module = _load_diffusion_benchmark_serving()

    req = module.RequestFuncInput(
        prompt="prompt",
        api_url="http://127.0.0.1:8098/v1/videos",
        model="default",
        num_frames=25,
    )
    args = SimpleNamespace(
        task="t2v",
        warmup_num_inference_steps=None,
        warmup_num_frames=None,
    )

    warm_req = module._make_warmup_request([req], 0, args)

    assert warm_req.num_frames == 1


def test_custom_dataset_reads_jsonl_without_pandas(tmp_path):
    module = _load_diffusion_benchmark_serving()

    dataset_path = tmp_path / "requests.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "prompt": "Floating crystal islands",
                "num_frames": 25,
                "fps": 16,
                "guidance_scale": 3.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        dataset_path=str(dataset_path),
        num_prompts=1,
        width=512,
        height=384,
        num_inference_steps=20,
        seed=42,
    )

    dataset = module.CustomDataset(args, "http://127.0.0.1:8098/v1/videos", "default")
    req = dataset[0]

    assert req.prompt == "Floating crystal islands"
    assert req.width == 512
    assert req.height == 384
    assert req.num_inference_steps == 20
    assert req.extra_body["num_frames"] == 25
    assert req.extra_body["fps"] == 16
    assert req.extra_body["guidance_scale"] == 3.0
