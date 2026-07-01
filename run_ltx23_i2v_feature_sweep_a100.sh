#!/usr/bin/env bash
set -euo pipefail

# LTX-2.3 I2V feature sweep for A100.
#
# This is a feature-support / feature-performance sweep, not a before/after A/B
# script. It measures the current checkout under the feature row:
#
#   LTX-2.3 I2V:
#     TeaCache: no
#     Cache-DiT: yes
#     SP Ulysses/Ring: yes
#     CFG-Parallel: yes
#     Tensor-Parallel: yes
#     Pipeline-Parallel: no
#     HSDP: no
#     CPU Offload Layerwise: yes
#     VAE-Patch-Parallel: yes, decode
#     Quantization: no
#     Step Execution: no
#
# Intended usage:
#   bash ./run_ltx23_i2v_feature_sweep_a100.sh
#
# Useful overrides:
#   NUM_PROMPTS=10 bash ./run_ltx23_i2v_feature_sweep_a100.sh
#   PROFILES="base cfg2 tp4 usp4" bash ./run_ltx23_i2v_feature_sweep_a100.sh
#   CASES="512x384x25 1024x576x81" bash ./run_ltx23_i2v_feature_sweep_a100.sh
#   bash ./run_ltx23_i2v_feature_sweep_a100.sh --summarize-only

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    DEFAULT_PYTHON="$ROOT_DIR/.venv/bin/python"
elif [[ -x /root/vllm-omni/.venv/bin/python ]]; then
    DEFAULT_PYTHON="/root/vllm-omni/.venv/bin/python"
else
    DEFAULT_PYTHON="python"
fi
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"

if [[ -d /data/models/Lightricks/LTX-2.3-Diffusers ]]; then
    DEFAULT_MODEL="/data/models/Lightricks/LTX-2.3-Diffusers"
else
    DEFAULT_MODEL="/data/models/Lightricks/LTX-2.3-diffusers"
fi

MODEL="${MODEL:-$DEFAULT_MODEL}"
MODEL_CLASS="${MODEL_CLASS:-LTX23ImageToVideoPipeline}"
TASK="${TASK:-i2v}"
HOST="${HOST:-127.0.0.1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/results/ltx23_i2v_feature_sweep_a100}"
IMAGE_PATH="${IMAGE_PATH:-$ROOT_DIR/tmp/cherry_blossom.jpg}"
IMAGE_URL="${IMAGE_URL:-https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg}"

CASES="${CASES:-512x384x25}"
PROFILES="${PROFILES:-base cache_dit ulysses2 ring2 hybrid_usp2_ring2 cfg2 tp2 tp4 layerwise_offload vae_patch2_tp2}"

NUM_PROMPTS="${NUM_PROMPTS:-5}"
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
ENABLE_PIPELINE_PROFILER="${ENABLE_PIPELINE_PROFILER:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
SAVE_OUTPUTS="${SAVE_OUTPUTS:-0}"

PROMPT="${PROMPT:-Cherry blossoms swaying gently in the breeze with synchronized ambient sound}"
NEG_PROMPT="${NEG_PROMPT:-worst quality, inconsistent motion, blurry, jittery, distorted}"
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
    if [[ "$PYTHON" == */* && ! -x "$PYTHON" ]]; then
        echo "Python executable not found: $PYTHON" >&2
        exit 1
    fi
    if [[ "$PYTHON" != */* ]] && ! command -v "$PYTHON" >/dev/null 2>&1; then
        echo "Python executable not found in PATH: $PYTHON" >&2
        exit 1
    fi
}

ensure_image() {
    if [[ "$TASK" != "i2v" && "$TASK" != "ti2v" ]]; then
        return 0
    fi
    if [[ -f "$IMAGE_PATH" ]]; then
        return 0
    fi

    mkdir -p "$(dirname "$IMAGE_PATH")"
    echo "[Image] $IMAGE_PATH not found; downloading $IMAGE_URL"
    if command -v curl >/dev/null 2>&1; then
        curl -LfsS "$IMAGE_URL" -o "$IMAGE_PATH"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$IMAGE_PATH" "$IMAGE_URL"
    else
        echo "Neither curl nor wget is available; provide IMAGE_PATH manually." >&2
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

    "$PYTHON" - "$dataset_file" "$width" "$height" "$frames" "$TASK" "$IMAGE_PATH" <<'PY'
import json
import os
import sys

path = sys.argv[1]
width = int(sys.argv[2])
height = int(sys.argv[3])
frames = int(sys.argv[4])
task = sys.argv[5]
image_path = sys.argv[6]

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
if task in {"i2v", "ti2v"}:
    record["image_paths"] = [image_path]

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
        cache_dit)
            SERVER_CMD+=(--cache-backend cache_dit)
            ;;
        ulysses2|usp2)
            SERVER_CMD+=(--usp 2)
            ;;
        ulysses4|usp4)
            SERVER_CMD+=(--usp 4)
            ;;
        ring2)
            SERVER_CMD+=(--ring 2)
            ;;
        ring4)
            SERVER_CMD+=(--ring 4)
            ;;
        hybrid_usp2_ring2|usp2_ring2)
            SERVER_CMD+=(--usp 2 --ring 2)
            ;;
        cfg2)
            SERVER_CMD+=(--cfg-parallel-size 2)
            ;;
        tp2)
            SERVER_CMD+=(--tensor-parallel-size 2)
            ;;
        tp4)
            SERVER_CMD+=(--tensor-parallel-size 4)
            ;;
        layerwise_offload)
            SERVER_CMD+=(--enable-layerwise-offload)
            ;;
        vae_patch2_tp2)
            SERVER_CMD+=(--tensor-parallel-size 2 --vae-patch-parallel-size 2 --vae-use-tiling)
            ;;
        vae_patch4_tp4)
            SERVER_CMD+=(--tensor-parallel-size 4 --vae-patch-parallel-size 4 --vae-use-tiling)
            ;;
        cache_dit_usp2)
            SERVER_CMD+=(--cache-backend cache_dit --usp 2)
            ;;
        cache_dit_cfg2)
            SERVER_CMD+=(--cache-backend cache_dit --cfg-parallel-size 2)
            ;;
        usp2_cfg2)
            SERVER_CMD+=(--usp 2 --cfg-parallel-size 2)
            ;;
        usp2_ring2_cfg2)
            SERVER_CMD+=(--usp 2 --ring 2 --cfg-parallel-size 2)
            ;;
        *)
            echo "Unknown profile: $profile" >&2
            echo "Supported profiles:" >&2
            echo "  base cache_dit ulysses2 ulysses4 ring2 ring4 hybrid_usp2_ring2 cfg2 tp2 tp4" >&2
            echo "  layerwise_offload vae_patch2_tp2 vae_patch4_tp4 cache_dit_usp2 cache_dit_cfg2" >&2
            echo "  usp2_cfg2 usp2_ring2_cfg2" >&2
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
    local save_dir="$6"

    local bench_cmd=(
        "$PYTHON" "$ROOT_DIR/benchmarks/diffusion/diffusion_benchmark_serving.py"
        --host "$HOST"
        --port "$port"
        --model "$MODEL"
        --endpoint /v1/videos
        --dataset custom
        --dataset-path "$dataset_file"
        --task "$TASK"
        --num-prompts "$prompts"
        --max-concurrency "$MAX_CONCURRENCY"
        --warmup-requests 0
        --disable-tqdm
        --output-file "$output_file"
    )
    if [[ "$SAVE_OUTPUTS" == "1" ]]; then
        bench_cmd+=(--save-dir "$save_dir")
    fi

    "${bench_cmd[@]}" 2>&1 | tee "$log_file"
}

summarize_results() {
    "$PYTHON" - "$OUTPUT_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
raw_dir = root / "raw"
summary = root / "summary.tsv"

def get_any(data, keys, default=""):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default

def stage_value(data, stat):
    for direct in (f"stage_0_gen_ms_{stat}", f"stage_0_{stat}"):
        if data.get(direct) is not None:
            return data[direct]

    stage_durations = data.get(f"stage_durations_{stat}") or {}
    if isinstance(stage_durations, dict):
        for key in ("stage_0_gen_ms", "stage_0", "0", 0):
            if stage_durations.get(key) is not None:
                return stage_durations[key]
    return ""

rows = []
for path in sorted(raw_dir.glob("*.json")):
    if path.name.startswith("warmup__"):
        continue
    parts = path.stem.split("__", 2)
    if len(parts) != 3:
        continue
    task, profile, case = parts
    with path.open() as f:
        data = json.load(f)
    completed = get_any(data, ["completed", "completed_requests", "num_completed_requests"], 0)
    failed = get_any(data, ["failed", "failed_requests", "num_failed_requests"], 0)
    try:
        status = "PASS" if int(completed) > 0 and int(failed) == 0 else "FAIL"
    except Exception:
        status = ""
    rows.append({
        "status": status,
        "task": task,
        "profile": profile,
        "case": case,
        "stage_0_gen_ms_mean": stage_value(data, "mean"),
        "stage_0_gen_ms_p50": stage_value(data, "p50"),
        "stage_0_gen_ms_p99": stage_value(data, "p99"),
        "latency_mean_s": get_any(data, ["latency_mean"]),
        "latency_p50_s": get_any(data, ["latency_p50", "latency_median"]),
        "latency_p99_s": get_any(data, ["latency_p99"]),
        "throughput_qps": get_any(data, ["throughput_qps"]),
        "peak_memory_mb_mean": get_any(data, ["peak_memory_mb_mean", "peak_memory_mb", "avg_peak_memory_mb"]),
        "peak_memory_mb_max": get_any(data, ["peak_memory_mb_max", "max_peak_memory_mb"]),
        "completed": completed,
        "failed": failed,
        "json": str(path),
    })

fieldnames = [
    "status",
    "task",
    "profile",
    "case",
    "stage_0_gen_ms_mean",
    "stage_0_gen_ms_p50",
    "stage_0_gen_ms_p99",
    "latency_mean_s",
    "latency_p50_s",
    "latency_p99_s",
    "throughput_qps",
    "peak_memory_mb_mean",
    "peak_memory_mb_max",
    "completed",
    "failed",
    "json",
]
with summary.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

print(f"[Summary] {summary}")
PY
}

start_server() {
    local profile="$1"
    local port="$2"
    local log_file="$3"

    SERVER_CMD=(
        "$PYTHON" -m vllm_omni.entrypoints.cli.main serve "$MODEL"
        --omni
        --host "$HOST"
        --port "$port"
        --model-class-name "$MODEL_CLASS"
    )

    if [[ "$ENFORCE_EAGER" == "1" ]]; then
        SERVER_CMD+=(--enforce-eager)
    fi
    if [[ "$ENABLE_PIPELINE_PROFILER" == "1" ]]; then
        SERVER_CMD+=(--enable-diffusion-pipeline-profiler)
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
    local profile="$1"
    local case_spec="$2"
    local width height frames case_label cell port

    IFS=x read -r width height frames <<<"$case_spec"
    case_label="${width}x${height}_${frames}f"
    cell="${TASK}__${profile}__${case_label}"
    port="$(open_port)"

    local dataset_file="$OUTPUT_ROOT/datasets/${TASK}__${case_label}.jsonl"
    local warmup_json="$OUTPUT_ROOT/raw/warmup__${cell}.json"
    local measured_json="$OUTPUT_ROOT/raw/${cell}.json"
    local warmup_log="$OUTPUT_ROOT/logs/warmup__${cell}.log"
    local measured_log="$OUTPUT_ROOT/logs/${cell}.log"
    local server_log="$OUTPUT_ROOT/logs/server__${cell}.log"
    local save_dir="$OUTPUT_ROOT/media/${cell}"

    make_dataset "$dataset_file" "$width" "$height" "$frames"

    if [[ "$SKIP_EXISTING" == "1" && -f "$measured_json" ]]; then
        echo "[Skip existing] $measured_json"
        summarize_results
        return 0
    fi

    echo
    echo "===== $cell ====="
    echo "dataset: $dataset_file"
    echo "raw json: $measured_json"

    start_server "$profile" "$port" "$server_log"
    wait_for_server "$port" "$server_log"

    if (( WARMUP_PROMPTS > 0 )); then
        echo "[Warmup] prompts=$WARMUP_PROMPTS"
        run_benchmark "$port" "$dataset_file" "$WARMUP_PROMPTS" "$warmup_json" "$warmup_log" "$save_dir/warmup"
    fi

    echo "[Measured] prompts=$NUM_PROMPTS"
    run_benchmark "$port" "$dataset_file" "$NUM_PROMPTS" "$measured_json" "$measured_log" "$save_dir/measured"

    cleanup_server
    summarize_results
}

main() {
    require_python
    ensure_image
    mkdir -p "$OUTPUT_ROOT"/{datasets,logs,raw,media}

    if [[ "${1:-}" == "--summarize-only" ]]; then
        summarize_results
        return 0
    fi

    echo "Output root: $OUTPUT_ROOT"
    echo "Task:        $TASK"
    echo "Model:       $MODEL"
    echo "Model class: $MODEL_CLASS"
    echo "Repo:        $ROOT_DIR"
    echo "Commit:      $(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || true)"
    echo "Image:       $IMAGE_PATH"
    echo "Cases:       $CASES"
    echo "Profiles:    $PROFILES"
    echo "Runs/cell:   warmup=$WARMUP_PROMPTS measured=$NUM_PROMPTS steps=$NUM_INFERENCE_STEPS"
    echo "Mode:        $([[ "$ENFORCE_EAGER" == "1" ]] && echo eager || echo compile/default)"
    echo

    local profile case_spec
    for profile in $PROFILES; do
        for case_spec in $CASES; do
            run_cell "$profile" "$case_spec"
        done
    done

    summarize_results
}

main "$@"
