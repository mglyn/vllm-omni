#!/usr/bin/env bash
set -euo pipefail

# Sweep LTX-2.3 transformer attention backends on a real serving pipeline.
#
# Matrix:
#   attention: baseline platform default, all SDPA, only text cross-attn SDPA
#   execution: eager, torch.compile
#   shapes:    512x384x25f, 1024x576x81f
#
# Each cell:
#   1. starts a fresh vLLM-Omni server,
#   2. runs one same-shape benchmark request as warmup,
#   3. starts torch profiler,
#   4. runs measured benchmark request(s),
#   5. stops profiler and gzips the trace via OmniTorchProfilerWrapper.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
MODEL="${MODEL:-/data/models/Lightricks/LTX-2.3-diffusers}"
MODEL_CLASS="${MODEL_CLASS:-LTX23Pipeline}"
HOST="${HOST:-127.0.0.1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/results/ltx23_attn_backend_sweep}"

NUM_PROMPTS="${NUM_PROMPTS:-1}"
WARMUP_PROMPTS="${WARMUP_PROMPTS:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
FPS="${FPS:-24}"
SEED="${SEED:-42}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
SERVER_TIMEOUT_S="${SERVER_TIMEOUT_S:-1800}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"

ATTN_MODES="${ATTN_MODES:-baseline text_cross_sdpa all_sdpa}"
EXEC_MODES="${EXEC_MODES:-eager compile}"
CASES="${CASES:-512x384x25 1024x576x81}"

# Keep this off by default: the diffusion pipeline profiler adds synchronizes
# and can distort the timeline. Enable only if stage_0_gen_ms is required.
ENABLE_PIPELINE_PROFILER="${ENABLE_PIPELINE_PROFILER:-0}"

PROMPT="${PROMPT:-Floating crystal islands in cosmic starry sky, glowing nebula, soft luminous particles flowing around, slow camera rotation}"
NEG_PROMPT="${NEG_PROMPT:-low quality, blurry, noise, watermark, text, deformed figures, cartoon style, over-saturated color, frame jump}"
export PROMPT NEG_PROMPT NUM_INFERENCE_STEPS FPS SEED

TEXT_CROSS_SDPA_CONFIG='{"default":{"backend":"FLASH_ATTN"},"per_role":{"ltx2.video_text_cross":{"backend":"TORCH_SDPA"},"ltx2.audio_text_cross":{"backend":"TORCH_SDPA"}}}'

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

json_stage_overrides() {
    local trace_dir="$1"
    "$PYTHON" - "$trace_dir" <<'PY'
import json
import sys

trace_dir = sys.argv[1]
cfg = {
    "0": {
        "profiler_config": {
            "profiler": "torch",
            "torch_profiler_dir": trace_dir,
            "torch_profiler_record_shapes": True,
            "torch_profiler_with_stack": True,
            "torch_profiler_with_memory": False,
            "torch_profiler_with_flops": False,
            "torch_profiler_use_gzip": True,
        }
    }
}
print(json.dumps(cfg, separators=(",", ":")))
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

post_profile() {
    local port="$1"
    local action="$2"
    curl -fsS \
        -X POST \
        -H "Content-Type: application/json" \
        -d '{"stages":[0]}' \
        "http://$HOST:$port/$action" >/dev/null
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

path, width, height, frames = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
prompt = os.environ["PROMPT"]
negative_prompt = os.environ["NEG_PROMPT"]
steps = int(os.environ["NUM_INFERENCE_STEPS"])
fps = int(os.environ["FPS"])
seed = int(os.environ["SEED"])
record = {
    "prompt": prompt,
    "negative_prompt": negative_prompt,
    "width": width,
    "height": height,
    "num_frames": frames,
    "fps": fps,
    "num_inference_steps": steps,
    "seed": seed,
}
with open(path, "w", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
PY
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
import sys
from pathlib import Path

root = Path(sys.argv[1])
raw_dir = root / "raw"
summary = root / "summary.tsv"
rows = []
for path in sorted(raw_dir.glob("*.json")):
    if path.name.startswith("warmup_"):
        continue
    with path.open() as f:
        data = json.load(f)
    name = path.stem
    exec_mode = ""
    attention = ""
    case = ""
    for candidate in ("eager", "compile"):
        marker = f"_{candidate}_"
        if marker in name:
            attention, case = name.split(marker, 1)
            exec_mode = candidate
            break
    if not attention:
        continue
    rows.append(
        {
            "attention": attention,
            "exec": exec_mode,
            "case": case,
            "stage_0_gen_ms_mean": data.get("stage_0_gen_ms_mean", ""),
            "stage_0_gen_ms_p50": data.get("stage_0_gen_ms_p50", ""),
            "stage_0_gen_ms_p99": data.get("stage_0_gen_ms_p99", ""),
            "serving_latency_mean_s": data.get("latency_mean", ""),
            "peak_memory_mb_mean": data.get("peak_memory_mb_mean", ""),
            "completed": data.get("completed_requests", ""),
            "failed": data.get("failed_requests", ""),
            "json": str(path),
        }
    )

fieldnames = [
    "attention",
    "exec",
    "case",
    "stage_0_gen_ms_mean",
    "stage_0_gen_ms_p50",
    "stage_0_gen_ms_p99",
    "serving_latency_mean_s",
    "peak_memory_mb_mean",
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

run_cell() {
    local attn_mode="$1"
    local exec_mode="$2"
    local case_spec="$3"

    local width height frames
    IFS=x read -r width height frames <<<"$case_spec"
    local case_label="${width}x${height}_${frames}f"
    local cell="${attn_mode}_${exec_mode}_${case_label}"
    local port
    port="$(open_port)"

    local cell_dir="$OUTPUT_ROOT/$cell"
    local trace_dir="$cell_dir/traces"
    local log_dir="$OUTPUT_ROOT/logs"
    local raw_dir="$OUTPUT_ROOT/raw"
    local dataset_dir="$OUTPUT_ROOT/datasets"
    mkdir -p "$trace_dir" "$log_dir" "$raw_dir" "$dataset_dir"

    local dataset_file="$dataset_dir/${case_label}.jsonl"
    make_dataset "$dataset_file" "$width" "$height" "$frames"

    local stage_overrides
    stage_overrides="$(json_stage_overrides "$trace_dir")"

    local -a server_cmd=(
        "$PYTHON" -m vllm_omni.entrypoints.cli.main serve "$MODEL"
        --omni
        --host "$HOST"
        --port "$port"
        --model-class-name "$MODEL_CLASS"
        --stage-overrides "$stage_overrides"
    )

    if [[ "$ENABLE_PIPELINE_PROFILER" == "1" ]]; then
        server_cmd+=(--enable-diffusion-pipeline-profiler)
    fi

    if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
        server_cmd+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
    fi

    case "$exec_mode" in
        eager)
            server_cmd+=(--enforce-eager)
            ;;
        compile)
            ;;
        *)
            echo "Unknown exec mode: $exec_mode" >&2
            exit 1
            ;;
    esac

    case "$attn_mode" in
        baseline)
            ;;
        all_sdpa)
            server_cmd+=(--diffusion-attention-backend TORCH_SDPA)
            ;;
        text_cross_sdpa)
            server_cmd+=(--diffusion-attention-config "$TEXT_CROSS_SDPA_CONFIG")
            ;;
        *)
            echo "Unknown attention mode: $attn_mode" >&2
            exit 1
            ;;
    esac

    local server_log="$log_dir/server_${cell}.log"
    echo
    echo "===== $cell ====="
    echo "server log: $server_log"
    echo "trace dir:  $trace_dir"
    echo "dataset:    $dataset_file"

    setsid env VLLM_WORKER_MULTIPROC_METHOD=spawn "${server_cmd[@]}" >"$server_log" 2>&1 &
    SERVER_PID=$!
    wait_for_server "$port" "$server_log"

    local warmup_json="$raw_dir/warmup_${cell}.json"
    local warmup_log="$log_dir/warmup_${cell}.log"
    echo "[Warmup] same shape, prompts=$WARMUP_PROMPTS"
    run_benchmark "$port" "$dataset_file" "$WARMUP_PROMPTS" "$warmup_json" "$warmup_log"

    echo "[Profile] start"
    post_profile "$port" start_profile

    local result_json="$raw_dir/${cell}.json"
    local result_log="$log_dir/bench_${cell}.log"
    echo "[Benchmark] measured prompts=$NUM_PROMPTS"
    run_benchmark "$port" "$dataset_file" "$NUM_PROMPTS" "$result_json" "$result_log"

    echo "[Profile] stop"
    post_profile "$port" stop_profile
    cleanup_server
}

main() {
    require_python
    mkdir -p "$OUTPUT_ROOT"

    echo "Output root: $OUTPUT_ROOT"
    echo "Model:       $MODEL"
    echo "Python:      $PYTHON"
    echo "Attention:   $ATTN_MODES"
    echo "Exec modes:  $EXEC_MODES"
    echo "Cases:       $CASES"
    echo "Runs/cell:   warmup=$WARMUP_PROMPTS measured=$NUM_PROMPTS steps=$NUM_INFERENCE_STEPS"
    echo "Profiler:    torch with stack, gzip, memory=false"
    if [[ "$ENABLE_PIPELINE_PROFILER" != "1" ]]; then
        echo "Stage ms:    disabled; set ENABLE_PIPELINE_PROFILER=1 if stage_0_gen_ms is required"
    fi

    for case_spec in $CASES; do
        local width height frames
        IFS=x read -r width height frames <<<"$case_spec"
        if (( width % 32 != 0 || height % 32 != 0 )); then
            echo "WARNING: case $case_spec is not divisible by 32; LTX/VAE may reject it." >&2
        fi
    done

    for attn_mode in $ATTN_MODES; do
        for exec_mode in $EXEC_MODES; do
            for case_spec in $CASES; do
                run_cell "$attn_mode" "$exec_mode" "$case_spec"
                summarize_results
            done
        done
    done

    summarize_results
}

main "$@"
