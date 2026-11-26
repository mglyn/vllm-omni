# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline BF16 GEMM tuning for a local CUDA or ROCm device. No model weights required."""

import argparse
import fcntl
import json
import math
import os
import random
import statistics
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
from vllm.triton_utils import triton

from vllm_omni.diffusion.layers.gemm_tuning.backends import BACKENDS, entry
from vllm_omni.diffusion.layers.gemm_tuning.profile import (
    OPERATION,
    Target,
    parse_shape,
    profile_entries,
    shape_key,
)
from vllm_omni.diffusion.layers.gemm_tuning.runtime import make_dispatch


def elapsed(fn, repeats):
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def sustained(fn, warm_ms, measure_ms):
    fn()  # Compile before recording events.
    torch.accelerator.synchronize()
    estimate = max(0.005, elapsed(fn, 3))
    elapsed(fn, max(1, math.ceil(warm_ms / estimate)))
    return elapsed(fn, max(1, math.ceil(measure_ms / estimate)))


def tune(shape, device, target, args):
    m, n, k = shape
    x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    weight = torch.randn(n, k, dtype=x.dtype, device=device)
    default = dispatch_unquantized_gemm()
    layer = torch.nn.Module()

    def baseline():
        return default(layer, x, weight, None)

    reference = baseline()
    rows = []
    ranked = []
    configs = [config for backend in BACKENDS.values() for config in backend.candidates(target, shape)]
    rng = random.Random(args.seed)
    rng.shuffle(configs)
    for config in configs:
        row = entry(config)
        rows.append(row)
        try:
            if not torch.equal(config.run(x, weight), reference):
                row["status"] = "output_mismatch"
                continue
            row["ms"] = sustained(lambda: config.run(x, weight), args.warm_ms, args.measure_ms)
        except (
            triton.runtime.errors.OutOfResources,
            torch.OutOfMemoryError,
            triton.compiler.errors.CompilationError,
            triton.runtime.errors.PTXASError,
        ) as error:
            row["status"] = f"{type(error).__name__}: {error}"
            continue
        row["status"] = "ok"
        ranked.append((row["ms"], config))
    result = {"screen": rows, "pairs_ms": [], "decision": {"backend": "default"}}
    if not ranked:
        return result
    _, config = min(ranked, key=lambda item: item[0])
    result["candidate"] = entry(config)
    # Validate the frozen candidate on independent operands with the same layout.
    x.normal_()
    weight.normal_()

    dispatch = make_dispatch(device, target, {shape: config}, default)

    def selected():
        return dispatch(layer, x, weight, None)

    if not torch.equal(selected(), baseline()):
        result["status"] = "holdout_output_mismatch"
        return result
    for i in range(args.pairs):
        pair = {}
        order = [("default", baseline), ("candidate", selected)]
        for name, fn in reversed(order) if i % 2 else order:
            pair[name] = sustained(fn, args.warm_ms, args.measure_ms)
        result["pairs_ms"].append(pair)
    gains = [100 * (1 - p["candidate"] / p["default"]) for p in result["pairs_ms"]]
    result["median_decrease_pct"] = statistics.median(gains)
    if result["median_decrease_pct"] >= args.min_gain and sum(g > 0 for g in gains) > len(gains) // 2:
        result["decision"] = entry(config)
    return result


def write_json(path, data):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def merge_decisions(directory, target, decisions):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / target.filename
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.is_file():
            data = json.loads(path.read_text())
            profile_entries(data, target)
        else:
            data = {"version": 1, "operation": OPERATION, "target": asdict(target), "entries": {}}
        data["entries"].update(decisions)
        write_json(path, data)
    return path


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", action="append", default=[], metavar="M,N,K")
    parser.add_argument("--trace", type=Path, nargs="+", default=[], help="Shape JSONL files from collect mode")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--report", type=Path, required=True, help="Raw screening, correctness and confirmation measurements"
    )
    parser.add_argument("--device", type=int, default=0, help="Local accelerator ordinal")
    parser.add_argument("--warm-ms", type=float, default=300)
    parser.add_argument("--measure-ms", type=float, default=150)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--min-gain", type=float, default=2)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    if (
        not all(math.isfinite(t) and t > 0 for t in (args.warm_ms, args.measure_ms))
        or args.pairs < 2
        or not 0 < args.min_gain < 100
    ):
        parser.error("Use positive durations, at least two holdout pairs, and a gain threshold between 0 and 100")
    keys = list(args.shape)
    for trace in args.trace:
        with trace.open() as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    if record.get("operation", OPERATION) != OPERATION:
                        parser.error(f"Unsupported GEMM operation in {trace}: {record['operation']!r}")
                    keys.append(record["shape"])
    shapes = sorted(set(map(parse_shape, keys)))
    if not shapes:
        parser.error("Supply at least one --shape or non-empty --trace")
    torch.accelerator.set_device_index(args.device)
    device = torch.device("cuda", args.device)  # PyTorch also uses the CUDA device type on ROCm.
    torch.manual_seed(args.seed)
    target = Target.current(device)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "target": asdict(target),
        "operation": OPERATION,
        "warm_ms": args.warm_ms,
        "measure_ms": args.measure_ms,
        "pairs": args.pairs,
        "min_gain_pct": args.min_gain,
        "seed": args.seed,
        "results": {},
    }
    with args.report.open("x") as stream:
        json.dump(results, stream)
    for shape in shapes:
        key = shape_key(shape)
        try:
            result = tune(shape, device, target, args)
        except torch.OutOfMemoryError:
            result = {"status": "skipped_out_of_memory"}
        results["results"][key] = result
        write_json(args.report, results)
        if "decision" in result:
            path = merge_decisions(args.output_dir, target, {key: result["decision"]})
            print(key, result["decision"], result.get("median_decrease_pct"), path, flush=True)
        else:
            print(key, result["status"], flush=True)


if __name__ == "__main__":
    main()
