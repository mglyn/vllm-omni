#!/usr/bin/env bash
set -euo pipefail

# LTX-2.3 mask-elision A/B sweep for A100.
#
# Intended usage:
#   # Run on the baseline commit / branch.
#   LABEL=before bash ./run_ltx23_mask_elision_sweep_a100.sh
#
#   # Run on the patched commit / branch, same OUTPUT_ROOT.
#   LABEL=after bash ./run_ltx23_mask_elision_sweep_a100.sh
#
# The script does not switch git branches. It measures the current checkout.
# Default scope is the small single-GPU compile A/B cell. Use stage_0_gen_ms as
# the primary metric. End-to-end serving latency includes polling / response
# overhead and is kept only as supporting context.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
MODEL="${MODEL:-/data/models/Lightricks/LTX-2.3-diffusers}"
MODEL_CLASS="${MODEL_CLASS:-LTX23Pipeline}"
HOST="${HOST:-127.0.0.1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/results/ltx23_mask_elision_sweep}"
LABEL="${LABEL:-$(git -C "$ROOT_DIR" branch --show-current 2>/dev/null || git -C "$ROOT_DIR" rev-parse --short HEAD)}"

CASES="${CASES:-512x384x25}"
EXEC_MODES="${EXEC_MODES:-compile}"
PROFILES="${PROFILES:-base}"

NUM_PROMPTS="${NUM_PROMPTS:-10}"
WARMUP_PROMPTS="${WARMUP_PROMPTS:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
FPS="${FPS:-24}"
FRAME_RATE="${FRAME_RATE:-24}"
AUDIO_SAMPLE_RATE="${AUDIO_SAMPLE_RATE:-48000}"
SEED="${SEED:-42}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
SERVER_TIMEOUT_S="${SERVER_TIMEOUT_S:-1800}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"

# Keep pipeline sub-stage profiling disabled by default. stage_0_gen_ms comes
# from stage metrics; the pipeline profiler adds per-module synchronization.
ENABLE_PIPELINE_PROFILER="${ENABLE_PIPELINE_PROFILER:-0}"
LTX23_DISABLE_MASK_ELISION="${VLLM_OMNI_LTX23_DISABLE_MASK_ELISION:-}"
if [[ -z "$LTX23_DISABLE_MASK_ELISION" ]]; then
    case "$LABEL" in
        before)
            LTX23_DISABLE_MASK_ELISION=1
            ;;
        after)
            LTX23_DISABLE_MASK_ELISION=0
            ;;
        *)
            LTX23_DISABLE_MASK_ELISION=0
            ;;
    esac
fi
export VLLM_OMNI_LTX23_DISABLE_MASK_ELISION="$LTX23_DISABLE_MASK_ELISION"

PROMPT="${PROMPT:-Floating crystal islands in cosmic starry sky, glowing nebula, soft luminous particles flowing around, slow camera rotation}"
NEG_PROMPT="${NEG_PROMPT:-low quality, blurry, noise, watermark, text, deformed figures, cartoon style, over-saturated color, frame jump}"
export PROMPT NEG_PROMPT NUM_INFERENCE_STEPS GUIDANCE_SCALE FPS FRAME_RATE AUDIO_SAMPLE_RATE SEED

SERVER_PID=""

cleanup_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            kill -TERM "-$SERVER_PID" 2>/dev/null || true
            sleep 3
        fi
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            kill -KILL "-$SERVER_PID" 2>/dev/null || true
        fi
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}
trap cleanup_server EXIT

require_python() {
    if [[ ! -x "$PYTHON" ]]; then
        echo "Python executable not found: $PYTHON" >&2
        exit 1
    fi
}

open_port() {
    "$PYTHON" - <<'PY'
import socket

s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

wait_for_server() {
    local port="$1"
    local log_file="$2"
    local deadline=$((SECONDS + SERVER_TIMEOUT_S))

    while (( SECONDS < deadline )); do
        if curl -fsS "http://$HOST:$port/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server exited before becoming ready. Log: $log_file" >&2
            tail -n 200 "$log_file" >&2 || true
            return 1
        fi
        sleep 2
    done

    echo "Timed out waiting for server. Log: $log_file" >&2
    tail -n 200 "$log_file" >&2 || true
    return 1
}

make_dataset() {
    local dataset_file="$1"
    local width="$2"
    local height="$3"
    local frames="$4"
    "$PYTHON" - "$dataset_file" "$width" "$height" "$frames" <<'PY'
import json
import os
import sys

path = sys.argv[1]
width = int(sys.argv[2])
height = int(sys.argv[3])
frames = int(sys.argv[4])
record = {
    "prompt": os.environ["PROMPT"],
    "negative_prompt": os.environ["NEG_PROMPT"],
    "width": width,
    "height": height,
    "num_frames": frames,
    "fps": int(os.environ["FPS"]),
    "frame_rate": int(os.environ["FRAME_RATE"]),
    "audio_sample_rate": int(os.environ["AUDIO_SAMPLE_RATE"]),
    "num_inference_steps": int(os.environ["NUM_INFERENCE_STEPS"]),
    "guidance_scale": float(os.environ["GUIDANCE_SCALE"]),
    "seed": int(os.environ["SEED"]),
}
with open(path, "w", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
PY
}

append_extra_server_args() {
    local words="$1"
    if [[ -n "$words" ]]; then
        # shellcheck disable=SC2206
        local extra=( $words )
        SERVER_CMD+=("${extra[@]}")
    fi
}

append_profile_server_args() {
    local profile="$1"
    case "$profile" in
        base)
            ;;
        cfg2)
            SERVER_CMD+=(--cfg-parallel-size 2)
            ;;
        usp2_cfg2)
            SERVER_CMD+=(--usp 2 --cfg-parallel-size 2)
            ;;
        usp4)
            SERVER_CMD+=(--usp 4)
            ;;
        tp2)
            SERVER_CMD+=(--tensor-parallel-size 2)
            ;;
        tp4)
            SERVER_CMD+=(--tensor-parallel-size 4)
            ;;
        *)
            echo "Unknown profile: $profile" >&2
            echo "Supported profiles: base cfg2 usp2_cfg2 usp4 tp2 tp4" >&2
            exit 1
            ;;
    esac
}

run_benchmark() {
    local port="$1"
    local dataset_file="$2"
    local prompts="$3"
    local output_file="$4"
    local log_file="$5"

    "$PYTHON" "$ROOT_DIR/benchmarks/diffusion/diffusion_benchmark_serving.py" \
        --host "$HOST" \
        --port "$port" \
        --model "$MODEL" \
        --endpoint /v1/videos \
        --dataset custom \
        --dataset-path "$dataset_file" \
        --task t2v \
        --num-prompts "$prompts" \
        --max-concurrency "$MAX_CONCURRENCY" \
        --warmup-requests 0 \
        --disable-tqdm \
        --output-file "$output_file" \
        2>&1 | tee "$log_file"
}

summarize_results() {
    "$PYTHON" - "$OUTPUT_ROOT" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
raw_dir = root / "raw"
summary = root / "summary.tsv"
comparison = root / "comparison.tsv"

def get_any(data, keys, default=""):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default

def stage_value(data, stat):
    candidates = [
        f"stage_0_gen_ms_{stat}",
        f"stage_0_{stat}",
    ]
    for key in candidates:
        if key in data and data[key] is not None:
            return data[key]

    stage_durations = data.get(f"stage_durations_{stat}") or {}
    if isinstance(stage_durations, dict) and stage_durations.get("stage_0_gen_ms") is not None:
        return stage_durations["stage_0_gen_ms"]

    stage_stats = data.get("stage_stats") or data.get("stage_metrics") or {}
    for stage_key in ("0", 0, "stage_0"):
        stage = stage_stats.get(stage_key) if isinstance(stage_stats, dict) else None
        if isinstance(stage, dict):
            for key in (f"gen_ms_{stat}", stat):
                if key in stage and stage[key] is not None:
                    return stage[key]

    by_stage = data.get("stage_durations") or data.get("stage_durations_ms") or {}
    if isinstance(by_stage, dict):
        stage = by_stage.get("0") or by_stage.get(0) or by_stage.get("stage_0")
        if isinstance(stage, dict):
            for key in (stat, f"gen_ms_{stat}"):
                if key in stage and stage[key] is not None:
                    return stage[key]
    return ""

def to_float(value):
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

rows = []
for path in sorted(raw_dir.glob("*.json")):
    if path.name.startswith("warmup__"):
        continue
    parts = path.stem.split("__", 3)
    if len(parts) != 4:
        continue
    label, exec_mode, profile, case = parts
    with path.open() as f:
        data = json.load(f)
    rows.append({
        "label": label,
        "exec": exec_mode,
        "profile": profile,
        "case": case,
        "stage_0_gen_ms_mean": stage_value(data, "mean"),
        "stage_0_gen_ms_p50": stage_value(data, "p50"),
        "stage_0_gen_ms_p99": stage_value(data, "p99"),
        "peak_memory_mb_mean": get_any(data, [
            "peak_memory_mb_mean",
            "peak_memory_mb",
            "avg_peak_memory_mb",
        ]),
        "completed": get_any(data, [
            "completed",
            "completed_requests",
            "num_completed_requests",
        ]),
        "failed": get_any(data, [
            "failed",
            "failed_requests",
            "num_failed_requests",
        ]),
        "json": str(path),
    })

fieldnames = [
    "label",
    "exec",
    "profile",
    "case",
    "stage_0_gen_ms_mean",
    "stage_0_gen_ms_p50",
    "stage_0_gen_ms_p99",
    "peak_memory_mb_mean",
    "completed",
    "failed",
    "json",
]
with summary.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

by_key = {}
for row in rows:
    by_key.setdefault((row["exec"], row["profile"], row["case"]), {})[row["label"]] = row

compare_rows = []
for (exec_mode, profile, case), labels in sorted(by_key.items()):
    if "before" not in labels or "after" not in labels:
        continue
    before = to_float(labels["before"]["stage_0_gen_ms_mean"])
    after = to_float(labels["after"]["stage_0_gen_ms_mean"])
    if before is None or after is None or math.isclose(before, 0.0):
        delta = ""
        delta_pct = ""
    else:
        delta = after - before
        delta_pct = delta / before * 100.0
    before_mem = to_float(labels["before"]["peak_memory_mb_mean"])
    after_mem = to_float(labels["after"]["peak_memory_mb_mean"])
    mem_delta = "" if before_mem is None or after_mem is None else after_mem - before_mem
    compare_rows.append({
        "exec": exec_mode,
        "profile": profile,
        "case": case,
        "before_stage_0_gen_ms_mean": "" if before is None else before,
        "after_stage_0_gen_ms_mean": "" if after is None else after,
        "delta_ms": delta,
        "delta_pct": delta_pct,
        "before_peak_memory_mb_mean": "" if before_mem is None else before_mem,
        "after_peak_memory_mb_mean": "" if after_mem is None else after_mem,
        "peak_memory_delta_mb": mem_delta,
    })

if compare_rows:
    compare_fields = [
        "exec",
        "profile",
        "case",
        "before_stage_0_gen_ms_mean",
        "after_stage_0_gen_ms_mean",
        "delta_ms",
        "delta_pct",
        "before_peak_memory_mb_mean",
        "after_peak_memory_mb_mean",
        "peak_memory_delta_mb",
    ]
    with comparison.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=compare_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(compare_rows)

print(f"[Summary] {summary}")
if compare_rows:
    print(f"[Comparison] {comparison}")
PY
}

start_server() {
    local exec_mode="$1"
    local profile="$2"
    local port="$3"
    local log_file="$4"

    SERVER_CMD=(
        "$PYTHON" -m vllm_omni.entrypoints.cli.main serve "$MODEL"
        --omni
        --host "$HOST"
        --port "$port"
        --model-class-name "$MODEL_CLASS"
    )

    if [[ "$ENABLE_PIPELINE_PROFILER" == "1" ]]; then
        SERVER_CMD+=(--enable-diffusion-pipeline-profiler)
    fi
    if [[ "$exec_mode" == "eager" ]]; then
        SERVER_CMD+=(--enforce-eager)
    elif [[ "$exec_mode" != "compile" ]]; then
        echo "Unknown exec mode: $exec_mode" >&2
        exit 1
    fi
    if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
        SERVER_CMD+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
    fi

    append_profile_server_args "$profile"
    append_extra_server_args "$EXTRA_SERVER_ARGS"

    echo "[Server] ${SERVER_CMD[*]}"
    echo "[Server log] $log_file"
    setsid "${SERVER_CMD[@]}" >"$log_file" 2>&1 &
    SERVER_PID="$!"
}

run_cell() {
    local exec_mode="$1"
    local profile="$2"
    local case_spec="$3"
    local width height frames case_label cell port

    IFS=x read -r width height frames <<<"$case_spec"
    case_label="${width}x${height}_${frames}f"
    cell="${LABEL}__${exec_mode}__${profile}__${case_label}"
    port="$(open_port)"

    local dataset_file="$OUTPUT_ROOT/datasets/${case_label}.jsonl"
    local warmup_json="$OUTPUT_ROOT/raw/warmup__${cell}.json"
    local measured_json="$OUTPUT_ROOT/raw/${cell}.json"
    local warmup_log="$OUTPUT_ROOT/logs/warmup__${cell}.log"
    local measured_log="$OUTPUT_ROOT/logs/${cell}.log"
    local server_log="$OUTPUT_ROOT/logs/server__${cell}.log"

    make_dataset "$dataset_file" "$width" "$height" "$frames"

    echo
    echo "===== $cell ====="
    echo "dataset: $dataset_file"
    echo "raw json: $measured_json"

    start_server "$exec_mode" "$profile" "$port" "$server_log"
    wait_for_server "$port" "$server_log"

    if (( WARMUP_PROMPTS > 0 )); then
        echo "[Warmup] prompts=$WARMUP_PROMPTS"
        run_benchmark "$port" "$dataset_file" "$WARMUP_PROMPTS" "$warmup_json" "$warmup_log"
    fi

    echo "[Measured] prompts=$NUM_PROMPTS"
    run_benchmark "$port" "$dataset_file" "$NUM_PROMPTS" "$measured_json" "$measured_log"

    cleanup_server
    summarize_results
}

main() {
    require_python
    mkdir -p "$OUTPUT_ROOT"/{datasets,logs,raw}

    if [[ "${1:-}" == "--summarize-only" ]]; then
        summarize_results
        return 0
    fi

    echo "Output root: $OUTPUT_ROOT"
    echo "Label:       $LABEL"
    echo "Model:       $MODEL"
    echo "Repo:        $ROOT_DIR"
    echo "Commit:      $(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || true)"
    echo "Cases:       $CASES"
    echo "Exec modes:  $EXEC_MODES"
    echo "Profiles:    $PROFILES"
    echo "Runs/cell:   warmup=$WARMUP_PROMPTS measured=$NUM_PROMPTS steps=$NUM_INFERENCE_STEPS"
    echo "Mask elision disabled: $VLLM_OMNI_LTX23_DISABLE_MASK_ELISION"
    echo

    local exec_mode profile case_spec
    for exec_mode in $EXEC_MODES; do
        for profile in $PROFILES; do
            for case_spec in $CASES; do
                run_cell "$exec_mode" "$profile" "$case_spec"
            done
        done
    done

    summarize_results
}

main "$@"
