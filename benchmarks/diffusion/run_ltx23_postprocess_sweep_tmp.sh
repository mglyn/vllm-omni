#!/usr/bin/env bash
set -euo pipefail

# Temporary LTX-2.3 postprocess sweep without pytest.
#
# This script starts vLLM-Omni servers directly and calls
# benchmarks/diffusion/diffusion_benchmark_serving.py directly.
#
# It runs before/after in the same checkout using VLLM_OMNI_LTX23_ASYNC_DTOH:
#   before: VLLM_OMNI_LTX23_ASYNC_DTOH=0
#   after:  VLLM_OMNI_LTX23_ASYNC_DTOH=1
# It runs both base and CFG2 profiles by default:
#   base: no extra parallel args
#   cfg2: --cfg-parallel-size 2
#
# Example:
#   MODEL=/data/models/Lightricks/LTX-2.3-Diffusers \
#   NUM_INFERENCE_STEPS=10 \
#   bash benchmarks/diffusion/run_ltx23_postprocess_sweep_tmp.sh
#
# To run only CFG2:
#   PROFILES=cfg2 bash benchmarks/diffusion/run_ltx23_postprocess_sweep_tmp.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"

MODEL="${MODEL:-/data/models/Lightricks/LTX-2.3-Diffusers}"
MODEL_CLASS_NAME="${MODEL_CLASS_NAME:-LTX23Pipeline}"
OUT_ROOT="${OUT_ROOT:-/root/results/ltx23_postprocess_benchmark_sweep}"
HOST="${HOST:-127.0.0.1}"

NUM_PROMPTS="${NUM_PROMPTS:-5}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
WARMUP_CONCURRENCY="${WARMUP_CONCURRENCY:-1}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-3}"
FPS="${FPS:-16}"
SEED="${SEED:-42}"
PROMPT="${PROMPT:-Floating crystal islands in cosmic starry sky, glowing nebula, soft luminous particles flowing around, slow camera rotation}"
NEG_PROMPT="${NEG_PROMPT:-low quality, blurry, noise, watermark, text, deformed figures, cartoon style, over-saturated color, frame jump}"

SMALL_WIDTH="${SMALL_WIDTH:-512}"
SMALL_HEIGHT="${SMALL_HEIGHT:-384}"
SMALL_FRAMES="${SMALL_FRAMES:-25}"
LARGE_WIDTH="${LARGE_WIDTH:-1024}"
LARGE_HEIGHT="${LARGE_HEIGHT:-576}"
LARGE_FRAMES="${LARGE_FRAMES:-81}"

# Optional JSON objects converted to CLI flags.
# Examples:
#   SERVER_EXTRA_ARGS_JSON='{"enable-layerwise-offload":true}'
#   BENCHMARK_EXTRA_PARAMS_JSON='{"request-rate":"inf","disable-tqdm":true}'
SERVER_EXTRA_ARGS_JSON="${SERVER_EXTRA_ARGS_JSON:-{}}"
BENCHMARK_EXTRA_PARAMS_JSON="${BENCHMARK_EXTRA_PARAMS_JSON:-{}}"
ENABLE_DIFFUSION_PIPELINE_PROFILER="${ENABLE_DIFFUSION_PIPELINE_PROFILER:-0}"

# Set RUN_BEFORE=0 or RUN_AFTER=0 to skip one side.
RUN_BEFORE="${RUN_BEFORE:-1}"
RUN_AFTER="${RUN_AFTER:-1}"
PROFILES="${PROFILES:-base cfg2}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/raw"

PYTHON_BIN=""
SERVER_PID=""

python_for_repo() {
  local repo="$1"
  if [[ -n "${PYTHON:-}" ]]; then
    printf '%s\n' "${PYTHON}"
  elif [[ -x "${repo}/.venv/bin/python" ]]; then
    printf '%s\n' "${repo}/.venv/bin/python"
  else
    printf '%s\n' "python"
  fi
}

json_to_cli_args() {
  local json_payload="$1"
  JSON_PAYLOAD="${json_payload}" "${PYTHON_BIN}" - <<'PY'
import json
import os

payload = json.loads(os.environ["JSON_PAYLOAD"])
if not isinstance(payload, dict):
    raise SystemExit("JSON payload must be an object")

for key, value in payload.items():
    flag = "--" + key
    if isinstance(value, bool):
        if value:
            print(flag)
    elif isinstance(value, (dict, list)):
        print(flag)
        print(json.dumps(value, separators=(",", ":")))
    elif value is not None:
        print(flag)
        print(str(value))
PY
}

get_free_port() {
  "${PYTHON_BIN}" - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("", 0))
    print(s.getsockname()[1])
PY
}

wait_for_port() {
  local host="$1"
  local port="$2"
  local pid="$3"
  "${PYTHON_BIN}" - "${host}" "${port}" "${pid}" <<'PY'
import os
import socket
import sys
import time

host, port, pid = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
deadline = time.time() + 1200
while time.time() < deadline:
    if pid > 0:
        try:
            os.kill(pid, 0)
        except OSError:
            raise SystemExit(f"server process {pid} exited before port became ready")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        if s.connect_ex((host, port)) == 0:
            raise SystemExit(0)
    time.sleep(2)
raise SystemExit(f"server did not start on {host}:{port} within timeout")
PY
}

stop_server() {
  if [[ -z "${SERVER_PID}" ]]; then
    return 0
  fi
  local pid="${SERVER_PID}"
  SERVER_PID=""
  "${PYTHON_BIN}" - "${pid}" <<'PY' || true
import os
import signal
import sys
import time

pid = int(sys.argv[1])
try:
    import psutil
except Exception:
    psutil = None

if psutil is not None:
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        raise SystemExit(0)
    children = parent.children(recursive=True)
    for proc in children:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        parent.terminate()
    except psutil.NoSuchProcess:
        pass
    gone, alive = psutil.wait_procs([parent, *children], timeout=20)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    raise SystemExit(0)

try:
    os.kill(pid, signal.SIGTERM)
except ProcessLookupError:
    raise SystemExit(0)
time.sleep(5)
try:
    os.kill(pid, signal.SIGKILL)
except ProcessLookupError:
    pass
PY
}

trap stop_server EXIT

start_server() {
  local label="$1"
  local mode="$2"
  local profile="$3"
  local async_dtoh="$4"
  local port="$5"
  local log_file="$6"

  local -a server_extra_args=()
  mapfile -t server_extra_args < <(json_to_cli_args "${SERVER_EXTRA_ARGS_JSON}")
  local -a profile_args=()
  case "${profile}" in
    base)
      ;;
    cfg2)
      profile_args+=(--cfg-parallel-size 2)
      ;;
    *)
      echo "Unknown profile: ${profile}" >&2
      exit 1
      ;;
  esac

  local -a server_args=(
    "${PYTHON_BIN}" -m vllm_omni.entrypoints.cli.main serve "${MODEL}"
    --omni
    --host "${HOST}"
    --port "${port}"
    --model-class-name "${MODEL_CLASS_NAME}"
  )
  if [[ "${ENABLE_DIFFUSION_PIPELINE_PROFILER}" == "1" ]]; then
    server_args+=(--enable-diffusion-pipeline-profiler)
  fi
  if [[ "${mode}" == "eager" ]]; then
    server_args+=(--enforce-eager)
  fi
  server_args+=("${profile_args[@]}")
  server_args+=("${server_extra_args[@]}")

  echo "[Server] ${label}/${mode}/${profile}: ${server_args[*]}"
  echo "[Server] log: ${log_file}"
  (
    cd "${REPO}"
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_OMNI_LTX23_ASYNC_DTOH="${async_dtoh}"
    exec "${server_args[@]}"
  ) >"${log_file}" 2>&1 &
  SERVER_PID=$!
  wait_for_port "${HOST}" "${port}" "${SERVER_PID}"
  echo "[Server] ready: ${HOST}:${port} pid=${SERVER_PID}"
}

run_case() {
  local label="$1"
  local mode="$2"
  local profile="$3"
  local port="$4"
  local case_name="$5"
  local width="$6"
  local height="$7"
  local frames="$8"

  local result_json="${OUT_ROOT}/raw/${label}_${mode}_${profile}_${case_name}.json"
  local log_file="${OUT_ROOT}/logs/${label}_${mode}_${profile}_${case_name}.log"

  local -a benchmark_extra_args=()
  mapfile -t benchmark_extra_args < <(json_to_cli_args "${BENCHMARK_EXTRA_PARAMS_JSON}")

  local -a bench_args=(
    "${PYTHON_BIN}" -u benchmarks/diffusion/diffusion_benchmark_serving.py
    --host "${HOST}"
    --port "${port}"
    --model "${MODEL}"
    --endpoint /v1/videos
    --dataset random
    --task t2v
    --num-prompts "${NUM_PROMPTS}"
    --max-concurrency "${MAX_CONCURRENCY}"
    --width "${width}"
    --height "${height}"
    --num-frames "${frames}"
    --fps "${FPS}"
    --num-inference-steps "${NUM_INFERENCE_STEPS}"
    --seed "${SEED}"
    --warmup-requests "${WARMUP_REQUESTS}"
    --warmup-concurrency "${WARMUP_CONCURRENCY}"
    --warmup-num-inference-steps "${NUM_INFERENCE_STEPS}"
    --warmup-num-frames "${frames}"
    --enable-negative-prompt
    --fixed-prompt "${PROMPT}"
    --fixed-negative-prompt "${NEG_PROMPT}"
    --output-file "${result_json}"
  )
  bench_args+=("${benchmark_extra_args[@]}")

  echo "[Benchmark] ${label}/${mode}/${profile}/${case_name}: ${bench_args[*]}"
  (
    cd "${REPO}"
    "${bench_args[@]}"
  ) 2>&1 | tee "${log_file}"
}

run_mode() {
  local label="$1"
  local async_dtoh="$2"
  local mode="$3"
  local profile="$4"
  local port
  port="$(get_free_port)"
  local server_log="${OUT_ROOT}/logs/server_${label}_${mode}_${profile}.log"

  echo
  echo "===== ${label}/${mode}/${profile} ====="
  echo "repo:    ${REPO}"
  echo "python:  ${PYTHON_BIN}"
  echo "results: ${OUT_ROOT}"
  echo "profile: ${profile}"
  echo "async DtoH env: VLLM_OMNI_LTX23_ASYNC_DTOH=${async_dtoh}"
  echo

  start_server "${label}" "${mode}" "${profile}" "${async_dtoh}" "${port}" "${server_log}"
  run_case "${label}" "${mode}" "${profile}" "${port}" "${SMALL_WIDTH}x${SMALL_HEIGHT}_${SMALL_FRAMES}f" "${SMALL_WIDTH}" "${SMALL_HEIGHT}" "${SMALL_FRAMES}"
  run_case "${label}" "${mode}" "${profile}" "${port}" "${LARGE_WIDTH}x${LARGE_HEIGHT}_${LARGE_FRAMES}f" "${LARGE_WIDTH}" "${LARGE_HEIGHT}" "${LARGE_FRAMES}"
  stop_server
}

summarize_results() {
  "${PYTHON_BIN}" - "${OUT_ROOT}" <<'PY'
import glob
import json
import os
import sys

out_root = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(out_root, "raw", "*.json"))):
    name = os.path.basename(path)[:-5]
    # label_mode_profile_case, where case itself contains underscores.
    parts = name.split("_", 3)
    if len(parts) == 4:
        label, mode, profile, case = parts
    else:
        label, mode, case = name.split("_", 2)
        profile = "base"
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    rows.append((label, mode, profile, case, result, path))

summary_path = os.path.join(out_root, "summary.tsv")
with open(summary_path, "w", encoding="utf-8") as out:
    out.write(
        "label\tmode\tprofile\tcase\tstage_0_gen_ms_mean\tstage_0_gen_ms_p50\t"
        "stage_0_gen_ms_p99\tthroughput_qps\tlatency_mean\tlatency_p50\t"
        "latency_p99\tpeak_memory_mb_mean\tcompleted\tfailed\tjson\n"
    )
    for label, mode, profile, case, result, path in rows:
        stage_mean = result.get("stage_durations_mean") or {}
        stage_p50 = result.get("stage_durations_p50") or {}
        stage_p99 = result.get("stage_durations_p99") or {}
        out.write(
            "\t".join(
                str(x)
                for x in [
                    label,
                    mode,
                    profile,
                    case,
                    stage_mean.get("stage_0_gen_ms", ""),
                    stage_p50.get("stage_0_gen_ms", ""),
                    stage_p99.get("stage_0_gen_ms", ""),
                    result.get("throughput_qps", ""),
                    result.get("latency_mean", ""),
                    result.get("latency_p50", ""),
                    result.get("latency_p99", ""),
                    result.get("peak_memory_mb_mean", ""),
                    result.get("completed_requests", result.get("completed", "")),
                    result.get("failed_requests", result.get("failed", "")),
                    path,
                ]
            )
            + "\n"
        )

print(f"[Summary] {summary_path}")
if rows:
    with open(summary_path, encoding="utf-8") as f:
        print(f.read(), end="")
PY
}

PYTHON_BIN="$(python_for_repo "${REPO}")"

echo "Output root: ${OUT_ROOT}"
echo "Model:       ${MODEL}"
echo "Repo:        ${REPO}"
echo "Cases:       ${SMALL_WIDTH}x${SMALL_HEIGHT}x${SMALL_FRAMES}f, ${LARGE_WIDTH}x${LARGE_HEIGHT}x${LARGE_FRAMES}f"
echo "Profiles:    ${PROFILES}"
echo "Runs/cell:   warmup=${WARMUP_REQUESTS}, measured=${NUM_PROMPTS}, steps=${NUM_INFERENCE_STEPS}"
echo "Prompt:      ${PROMPT}"
echo "Neg prompt:  ${NEG_PROMPT}"

if [[ "${RUN_BEFORE}" == "1" ]]; then
  for profile in ${PROFILES}; do
    run_mode before 0 eager "${profile}"
    run_mode before 0 compile "${profile}"
  done
fi

if [[ "${RUN_AFTER}" == "1" ]]; then
  for profile in ${PROFILES}; do
    run_mode after 1 eager "${profile}"
    run_mode after 1 compile "${profile}"
  done
fi

summarize_results
