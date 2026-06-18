#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Sweep LTX-2.3 async video postprocess A/B on online serving.
#
# Default matrix:
#   scenario: small, large
#   parallel: 1gpu
#   runtime : eager, compile
#   A/B     : sync postprocess, async postprocess
#
# Each cell starts a fresh server, runs one same-shape warmup request, then
# measures REQUESTS_PER_CELL requests with diffusion_benchmark_serving.py.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3}"
VLLM_CMD="${VLLM_CMD:-vllm}"
MODEL="${MODEL:-/data/models/Lightricks/LTX-2.3-Diffusers}"
MODEL_CLASS_NAME="${MODEL_CLASS_NAME:-LTX23Pipeline}"
OUT_DIR="${OUT_DIR:-results/ltx23_async_postprocess_ab_sweep/$(date +%Y%m%d_%H%M%S)}"

CUDA_VISIBLE_DEVICES_1GPU="${CUDA_VISIBLE_DEVICES_1GPU:-0}"
CUDA_VISIBLE_DEVICES_CFG2="${CUDA_VISIBLE_DEVICES_CFG2:-0,1}"
PORT_BASE="${PORT_BASE:-8098}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1800}"
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"
PARALLEL_MODES="${PARALLEL_MODES:-1gpu}"
RUNTIME_MODES="${RUNTIME_MODES:-eager compile}"

SMALL_WIDTH="${SMALL_WIDTH:-512}"
SMALL_HEIGHT="${SMALL_HEIGHT:-384}"
SMALL_NUM_FRAMES="${SMALL_NUM_FRAMES:-25}"
SMALL_FPS="${SMALL_FPS:-16}"
SMALL_NUM_INFERENCE_STEPS="${SMALL_NUM_INFERENCE_STEPS:-20}"
LARGE_WIDTH="${LARGE_WIDTH:-1024}"
LARGE_HEIGHT="${LARGE_HEIGHT:-576}"
LARGE_NUM_FRAMES="${LARGE_NUM_FRAMES:-81}"
LARGE_FPS="${LARGE_FPS:-16}"
LARGE_NUM_INFERENCE_STEPS="${LARGE_NUM_INFERENCE_STEPS:-10}"
SCENARIOS="${SCENARIOS:-small:${SMALL_WIDTH}:${SMALL_HEIGHT}:${SMALL_NUM_FRAMES}:${SMALL_NUM_INFERENCE_STEPS}:${SMALL_FPS} large:${LARGE_WIDTH}:${LARGE_HEIGHT}:${LARGE_NUM_FRAMES}:${LARGE_NUM_INFERENCE_STEPS}:${LARGE_FPS}}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.0}"
SEED="${SEED:-42}"
PROMPT="${PROMPT:-Floating crystal islands in cosmic starry sky, glowing nebula, soft luminous particles flowing around, slow camera rotation}"
NEG_PROMPT="${NEG_PROMPT:-low quality, blurry, noise, watermark, text, deformed figures, cartoon style, over-saturated color, frame jump}"

REQUESTS_PER_CELL="${REQUESTS_PER_CELL:-10}"
REPEATS_PER_CELL="${REPEATS_PER_CELL:-1}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
WARMUP_CONCURRENCY="${WARMUP_CONCURRENCY:-1}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
REQUEST_RATE="${REQUEST_RATE:-inf}"

mkdir -p "$OUT_DIR"

EXTRA_BODY="${EXTRA_BODY:-}"
if [[ -z "$EXTRA_BODY" ]]; then
    EXTRA_BODY="$(
        GUIDANCE_SCALE="$GUIDANCE_SCALE" NEG_PROMPT="$NEG_PROMPT" "$PYTHON" - <<'PY'
import json
import os

print(json.dumps({
    "guidance_scale": float(os.environ["GUIDANCE_SCALE"]),
    "negative_prompt": os.environ["NEG_PROMPT"],
}))
PY
    )"
fi

make_dataset() {
    local dataset_path="$1"
    local width="$2"
    local height="$3"
    local num_frames="$4"
    local steps="$5"
    local fps="$6"

    PROMPT="$PROMPT" WIDTH="$width" HEIGHT="$height" NUM_FRAMES="$num_frames" FPS="$fps" \
    NUM_INFERENCE_STEPS="$steps" SEED="$SEED" "$PYTHON" - "$dataset_path" <<'PY'
import json
import os
import sys

path = sys.argv[1]
record = {
    "prompt": os.environ["PROMPT"],
    "width": int(os.environ["WIDTH"]),
    "height": int(os.environ["HEIGHT"]),
    "num_inference_steps": int(os.environ["NUM_INFERENCE_STEPS"]),
    "seed": int(os.environ["SEED"]),
    "num_frames": int(os.environ["NUM_FRAMES"]),
    "fps": int(os.environ["FPS"]),
}
with open(path, "w", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")
PY
}

SERVER_PID=""

cleanup_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -- "-$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}
trap cleanup_server EXIT

wait_for_server() {
    local base_url="$1"
    local log_file="$2"
    local deadline=$((SECONDS + SERVER_READY_TIMEOUT))

    until curl -fsS "$base_url/health" >/dev/null 2>&1; do
        if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server exited before becoming healthy. Log tail:" >&2
            tail -200 "$log_file" >&2 || true
            return 1
        fi
        if (( SECONDS >= deadline )); then
            echo "Timed out waiting for $base_url. Log tail:" >&2
            tail -200 "$log_file" >&2 || true
            return 1
        fi
        sleep 2
    done
}

run_cell() {
    local scenario="$1"
    local width="$2"
    local height="$3"
    local num_frames="$4"
    local steps="$5"
    local fps="$6"
    local parallel="$7"
    local runtime="$8"
    local async_enabled="$9"
    local port="${10}"

    local async_label="sync"
    if [[ "$async_enabled" == "1" ]]; then
        async_label="async"
    fi

    local cell="${scenario}_${parallel}_${runtime}_${async_label}"
    local cell_dir="$OUT_DIR/$cell"
    local server_log="$cell_dir/server.log"
    local base_url="http://127.0.0.1:${port}"
    local dataset_path="$cell_dir/ltx23_prompt.jsonl"
    mkdir -p "$cell_dir"
    make_dataset "$dataset_path" "$width" "$height" "$num_frames" "$steps" "$fps"

    local devices
    local cfg_args=()
    if [[ "$parallel" == "cfg2" ]]; then
        devices="$CUDA_VISIBLE_DEVICES_CFG2"
        cfg_args=(--cfg-parallel-size 2)
    else
        devices="$CUDA_VISIBLE_DEVICES_1GPU"
    fi

    local runtime_args=()
    if [[ "$runtime" == "eager" ]]; then
        runtime_args=(--enforce-eager)
    fi

    read -r -a vllm_cmd <<< "$VLLM_CMD"
    read -r -a extra_server_args <<< "$SERVER_EXTRA_ARGS"

    local serve_cmd=(
        "${vllm_cmd[@]}" serve "$MODEL"
        --omni
        --port "$port"
        --model-class-name "$MODEL_CLASS_NAME"
        "${cfg_args[@]}"
        "${runtime_args[@]}"
        "${extra_server_args[@]}"
    )

    echo "=== Starting $cell on CUDA_VISIBLE_DEVICES=$devices port=$port ===" | tee "$cell_dir/cell.log"
    printf 'Server command: CUDA_VISIBLE_DEVICES=%q VLLM_OMNI_LTX23_ASYNC_VIDEO_POSTPROCESS=%q ' \
        "$devices" "$async_enabled" | tee -a "$cell_dir/cell.log"
    printf '%q ' "${serve_cmd[@]}" | tee -a "$cell_dir/cell.log"
    echo | tee -a "$cell_dir/cell.log"

    CUDA_VISIBLE_DEVICES="$devices" \
    VLLM_OMNI_LTX23_ASYNC_VIDEO_POSTPROCESS="$async_enabled" \
        setsid "${serve_cmd[@]}" >"$server_log" 2>&1 &
    SERVER_PID=$!

    wait_for_server "$base_url" "$server_log"

    for repeat in $(seq 1 "$REPEATS_PER_CELL"); do
        local metrics_file="$cell_dir/metrics_repeat_${repeat}.json"
        local bench_log="$cell_dir/benchmark_repeat_${repeat}.log"
        local warmup_requests="$WARMUP_REQUESTS"

        local bench_cmd=(
            "$PYTHON" benchmarks/diffusion/diffusion_benchmark_serving.py
            --base-url "$base_url"
            --endpoint /v1/videos
            --dataset custom
            --dataset-path "$dataset_path"
            --task t2v
            --model default
            --num-prompts "$REQUESTS_PER_CELL"
            --max-concurrency "$MAX_CONCURRENCY"
            --request-rate "$REQUEST_RATE"
            --width "$width"
            --height "$height"
            --num-frames "$num_frames"
            --fps "$fps"
            --num-inference-steps "$steps"
            --seed "$SEED"
            --warmup-requests "$warmup_requests"
            --warmup-num-inference-steps "$steps"
            --warmup-num-frames "$num_frames"
            --warmup-concurrency "$WARMUP_CONCURRENCY"
            --extra-body "$EXTRA_BODY"
            --output-file "$metrics_file"
            --disable-tqdm
        )

        echo "--- Benchmark $cell repeat $repeat/$REPEATS_PER_CELL ---" | tee -a "$cell_dir/cell.log"
        printf 'Benchmark command: ' | tee -a "$cell_dir/cell.log"
        printf '%q ' "${bench_cmd[@]}" | tee -a "$cell_dir/cell.log"
        echo | tee -a "$cell_dir/cell.log"

        "${bench_cmd[@]}" 2>&1 | tee "$bench_log"
    done

    cleanup_server
}

read -r -a parallel_modes <<< "$PARALLEL_MODES"
read -r -a runtime_modes <<< "$RUNTIME_MODES"
read -r -a scenario_specs <<< "$SCENARIOS"

cell_index=0
for scenario_spec in "${scenario_specs[@]}"; do
    IFS=: read -r scenario width height num_frames steps fps <<< "$scenario_spec"
    if [[ -z "$scenario" || -z "$width" || -z "$height" || -z "$num_frames" || -z "$steps" || -z "$fps" ]]; then
        echo "Invalid scenario spec: '$scenario_spec'. Expected name:width:height:num_frames:steps:fps" >&2
        exit 1
    fi
    for parallel in "${parallel_modes[@]}"; do
        for runtime in "${runtime_modes[@]}"; do
            for async_enabled in 0 1; do
                run_cell \
                    "$scenario" "$width" "$height" "$num_frames" "$steps" "$fps" \
                    "$parallel" "$runtime" "$async_enabled" "$((PORT_BASE + cell_index))"
                cell_index=$((cell_index + 1))
            done
        done
    done
done

"$PYTHON" - "$OUT_DIR" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
rows = []
for path in sorted(out_dir.glob("*/*metrics_repeat_*.json")):
    cell = path.parent.name
    scenario, parallel, runtime, variant = cell.split("_", 3)
    data = json.loads(path.read_text())
    stage_durations_mean = data.get("stage_durations_mean") or {}
    rows.append({
        "cell": cell,
        "scenario": scenario,
        "parallel": parallel,
        "runtime": runtime,
        "variant": variant,
        "repeat": path.stem.rsplit("_", 1)[-1],
        "throughput_qps": float(data.get("throughput_qps", 0.0)),
        "latency_mean": float(data.get("latency_mean", 0.0)),
        "latency_median": float(data.get("latency_median", 0.0)),
        "latency_p95": float(data.get("latency_p95", 0.0)),
        "latency_p99": float(data.get("latency_p99", 0.0)),
        "peak_memory_mb_max": float(data.get("peak_memory_mb_max", 0.0)),
        "stage_0_gen_ms": float(stage_durations_mean.get("stage_0_gen_ms", 0.0)),
        "queue_wait_ms": float(stage_durations_mean.get("queue_wait_ms", 0.0)),
        "completed_requests": int(data.get("completed_requests", 0)),
        "failed_requests": int(data.get("failed_requests", 0)),
    })

def mean(values):
    return statistics.mean(values) if values else 0.0

summary = {}
for row in rows:
    key = (row["scenario"], row["parallel"], row["runtime"], row["variant"])
    bucket = summary.setdefault(key, {k: [] for k in (
        "throughput_qps", "latency_mean", "latency_median",
        "latency_p95", "latency_p99", "peak_memory_mb_max",
        "stage_0_gen_ms", "queue_wait_ms",
    )})
    for metric in bucket:
        if metric in row:
            bucket[metric].append(row[metric])

summary_rows = []
for (scenario, parallel, runtime, variant), bucket in sorted(summary.items()):
    summary_rows.append({
        "scenario": scenario,
        "parallel": parallel,
        "runtime": runtime,
        "variant": variant,
        "throughput_qps_mean": mean(bucket["throughput_qps"]),
        "latency_mean_mean": mean(bucket["latency_mean"]),
        "latency_median_mean": mean(bucket["latency_median"]),
        "latency_p95_mean": mean(bucket["latency_p95"]),
        "latency_p99_mean": mean(bucket["latency_p99"]),
        "peak_memory_mb_max_mean": mean(bucket["peak_memory_mb_max"]),
        "stage_0_gen_ms_mean": mean(bucket["stage_0_gen_ms"]),
        "queue_wait_ms_mean": mean(bucket["queue_wait_ms"]),
    })

by_cell = {(r["scenario"], r["parallel"], r["runtime"], r["variant"]): r for r in summary_rows}
comparisons = []
scenario_names = sorted({r["scenario"] for r in summary_rows})
parallel_names = sorted({r["parallel"] for r in summary_rows})
runtime_names = sorted({r["runtime"] for r in summary_rows})
for scenario in scenario_names:
    for parallel in parallel_names:
        for runtime in runtime_names:
            before = by_cell.get((scenario, parallel, runtime, "sync"))
            after = by_cell.get((scenario, parallel, runtime, "async"))
            if not before or not after:
                continue
            before_latency = before["latency_mean_mean"]
            after_latency = after["latency_mean_mean"]
            before_stage0 = before["stage_0_gen_ms_mean"]
            after_stage0 = after["stage_0_gen_ms_mean"]
            comparisons.append({
                "scenario": scenario,
                "parallel": parallel,
                "runtime": runtime,
                "latency_mean_sync": before_latency,
                "latency_mean_async": after_latency,
                "latency_reduction_pct": (
                    (before_latency - after_latency) / before_latency * 100.0
                    if before_latency > 0 else 0.0
                ),
                "speedup_by_latency": (
                    before_latency / after_latency
                    if after_latency > 0 else 0.0
                ),
                "throughput_qps_sync": before["throughput_qps_mean"],
                "throughput_qps_async": after["throughput_qps_mean"],
                "speedup_by_throughput": (
                    after["throughput_qps_mean"] / before["throughput_qps_mean"]
                    if before["throughput_qps_mean"] > 0 else 0.0
                ),
                "stage_0_gen_ms_sync": before_stage0,
                "stage_0_gen_ms_async": after_stage0,
                "stage_0_gen_ms_reduction_pct": (
                    (before_stage0 - after_stage0) / before_stage0 * 100.0
                    if before_stage0 > 0 else 0.0
                ),
            })

(out_dir / "summary.json").write_text(json.dumps({
    "rows": rows,
    "summary": summary_rows,
    "comparisons": comparisons,
}, indent=2))

with (out_dir / "summary.csv").open("w", newline="") as f:
    fieldnames = [
        "scenario", "parallel", "runtime", "variant", "throughput_qps_mean",
        "latency_mean_mean", "latency_median_mean", "latency_p95_mean",
        "latency_p99_mean", "peak_memory_mb_max_mean",
        "stage_0_gen_ms_mean", "queue_wait_ms_mean",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)

with (out_dir / "comparisons.csv").open("w", newline="") as f:
    fieldnames = [
        "scenario", "parallel", "runtime", "latency_mean_sync", "latency_mean_async",
        "latency_reduction_pct", "speedup_by_latency",
        "throughput_qps_sync", "throughput_qps_async", "speedup_by_throughput",
        "stage_0_gen_ms_sync", "stage_0_gen_ms_async",
        "stage_0_gen_ms_reduction_pct",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(comparisons)

print(f"Wrote {out_dir / 'summary.json'}")
print(f"Wrote {out_dir / 'summary.csv'}")
print(f"Wrote {out_dir / 'comparisons.csv'}")
PY

echo "Sweep complete: $OUT_DIR"
