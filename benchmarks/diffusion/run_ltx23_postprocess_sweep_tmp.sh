#!/usr/bin/env bash
set -euo pipefail

# Temporary LTX-2.3 postprocess sweep.
#
# This script uses tests/dfx/perf/scripts/run_diffusion_benchmark.py, which
# starts a vLLM-Omni server and calls benchmarks/diffusion/diffusion_benchmark_serving.py.
#
# Expected setup:
#   BEFORE_REPO points at a worktree/checkout before the DtoH overlap change.
#   AFTER_REPO points at a worktree/checkout after the DtoH overlap change.
#
# Example:
#   BEFORE_REPO=/root/vllm-omni-before-postprocess \
#   AFTER_REPO=/root/vllm-omni \
#   MODEL=/data/models/Lightricks/LTX-2.3-Diffusers \
#   NUM_INFERENCE_STEPS=10 \
#   bash benchmarks/diffusion/run_ltx23_postprocess_sweep_tmp.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_AFTER_REPO="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

BEFORE_REPO="${BEFORE_REPO:-/root/vllm-omni-before-postprocess}"
AFTER_REPO="${AFTER_REPO:-${DEFAULT_AFTER_REPO}}"
MODEL="${MODEL:-/data/models/Lightricks/LTX-2.3-Diffusers}"
MODEL_CLASS_NAME="${MODEL_CLASS_NAME:-LTX23Pipeline}"
OUT_ROOT="${OUT_ROOT:-/root/results/ltx23_postprocess_benchmark_sweep}"

NUM_PROMPTS="${NUM_PROMPTS:-10}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
WARMUP_CONCURRENCY="${WARMUP_CONCURRENCY:-1}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-3}"
FPS="${FPS:-16}"
SEED="${SEED:-42}"

SMALL_WIDTH="${SMALL_WIDTH:-512}"
SMALL_HEIGHT="${SMALL_HEIGHT:-384}"
SMALL_FRAMES="${SMALL_FRAMES:-25}"
LARGE_WIDTH="${LARGE_WIDTH:-1024}"
LARGE_HEIGHT="${LARGE_HEIGHT:-576}"
LARGE_FRAMES="${LARGE_FRAMES:-81}"

# Optional JSON objects merged into every server / benchmark entry.
# Examples:
#   SERVER_EXTRA_ARGS_JSON='{"enable-layerwise-offload":true}'
#   BENCHMARK_EXTRA_PARAMS_JSON='{"request-rate":"inf"}'
SERVER_EXTRA_ARGS_JSON="${SERVER_EXTRA_ARGS_JSON:-{}}"
BENCHMARK_EXTRA_PARAMS_JSON="${BENCHMARK_EXTRA_PARAMS_JSON:-{}}"

# Set RUN_BEFORE=0 or RUN_AFTER=0 to skip one side.
RUN_BEFORE="${RUN_BEFORE:-1}"
RUN_AFTER="${RUN_AFTER:-1}"

export MODEL MODEL_CLASS_NAME
export NUM_PROMPTS WARMUP_REQUESTS WARMUP_CONCURRENCY MAX_CONCURRENCY
export NUM_INFERENCE_STEPS FPS SEED
export SMALL_WIDTH SMALL_HEIGHT SMALL_FRAMES
export LARGE_WIDTH LARGE_HEIGHT LARGE_FRAMES
export SERVER_EXTRA_ARGS_JSON BENCHMARK_EXTRA_PARAMS_JSON

mkdir -p "${OUT_ROOT}/configs"

python_for_repo() {
  local repo="$1"
  if [[ -n "${PYTHON:-}" ]]; then
    printf '%s\n' "${PYTHON}"
  elif [[ -x "${repo}/.venv/bin/python" ]]; then
    printf '%s\n' "${repo}/.venv/bin/python"
  elif [[ -x "${AFTER_REPO}/.venv/bin/python" ]]; then
    printf '%s\n' "${AFTER_REPO}/.venv/bin/python"
  else
    printf '%s\n' "python"
  fi
}

write_config() {
  local label="$1"
  local path="$2"
  LABEL="${label}" CONFIG_PATH="${path}" python - <<'PY'
import json
import os

label = os.environ["LABEL"]
path = os.environ["CONFIG_PATH"]
model = os.environ["MODEL"]
model_class_name = os.environ["MODEL_CLASS_NAME"]
num_prompts = int(os.environ["NUM_PROMPTS"])
warmup_requests = int(os.environ["WARMUP_REQUESTS"])
warmup_concurrency = int(os.environ["WARMUP_CONCURRENCY"])
max_concurrency = int(os.environ["MAX_CONCURRENCY"])
steps = int(os.environ["NUM_INFERENCE_STEPS"])
fps = int(os.environ["FPS"])
seed = int(os.environ["SEED"])
server_extra = json.loads(os.environ["SERVER_EXTRA_ARGS_JSON"])
benchmark_extra = json.loads(os.environ["BENCHMARK_EXTRA_PARAMS_JSON"])

cases = [
    {
        "name": f"{os.environ['SMALL_WIDTH']}x{os.environ['SMALL_HEIGHT']}_{os.environ['SMALL_FRAMES']}f_steps{steps}",
        "width": int(os.environ["SMALL_WIDTH"]),
        "height": int(os.environ["SMALL_HEIGHT"]),
        "num-frames": int(os.environ["SMALL_FRAMES"]),
    },
    {
        "name": f"{os.environ['LARGE_WIDTH']}x{os.environ['LARGE_HEIGHT']}_{os.environ['LARGE_FRAMES']}f_steps{steps}",
        "width": int(os.environ["LARGE_WIDTH"]),
        "height": int(os.environ["LARGE_HEIGHT"]),
        "num-frames": int(os.environ["LARGE_FRAMES"]),
    },
]

def bench_entry(case):
    entry = {
        "name": case["name"],
        "dataset": "random",
        "task": "t2v",
        "width": case["width"],
        "height": case["height"],
        "num-frames": case["num-frames"],
        "fps": fps,
        "num-inference-steps": steps,
        "num-prompts": num_prompts,
        "max-concurrency": max_concurrency,
        "warmup-requests": warmup_requests,
        "warmup-concurrency": warmup_concurrency,
        "warmup-num-inference-steps": steps,
        "seed": seed,
        "enable-negative-prompt": True,
        "skip-performance-assertion": True,
    }
    entry.update(benchmark_extra)
    return entry

def server(mode):
    serve_args = {
        "model-class-name": model_class_name,
        "enable-diffusion-pipeline-profiler": True,
    }
    if mode == "eager":
        serve_args["enforce-eager"] = True
    serve_args.update(server_extra)
    return {
        "test_name": f"ltx23_{label}_{mode}",
        "description": f"LTX-2.3 {label} {mode} postprocess sweep",
        "server_type": "vllm-omni",
        "benchmark_endpoint": "/v1/videos",
        "server_params": {
            "model": model,
            "serve_args": serve_args,
        },
        "benchmark_params": [bench_entry(case) for case in cases],
    }

configs = [server("eager"), server("compile")]
with open(path, "w", encoding="utf-8") as f:
    json.dump(configs, f, indent=2, ensure_ascii=False)
print(path)
PY
}

run_side() {
  local label="$1"
  local repo="$2"

  if [[ ! -d "${repo}" ]]; then
    echo "[Skip] ${label}: repo does not exist: ${repo}" >&2
    return 0
  fi
  if [[ ! -f "${repo}/tests/dfx/perf/scripts/run_diffusion_benchmark.py" ]]; then
    echo "[Skip] ${label}: benchmark runner not found in ${repo}" >&2
    return 0
  fi

  local config="${OUT_ROOT}/configs/ltx23_${label}_postprocess_sweep.json"
  write_config "${label}" "${config}"

  local result_dir="${OUT_ROOT}/${label}"
  mkdir -p "${result_dir}"

  local py
  py="$(python_for_repo "${repo}")"

  echo
  echo "===== ${label} ====="
  echo "repo:    ${repo}"
  echo "python:  ${py}"
  echo "config:  ${config}"
  echo "results: ${result_dir}"
  echo

  (
    cd "${repo}"
    DIFFUSION_BENCHMARK_DIR="${result_dir}" \
      "${py}" -m pytest -s tests/dfx/perf/scripts/run_diffusion_benchmark.py \
        --test-config-file "${config}" \
        --tb=short \
        --disable-warnings
  )

  summarize_results "${label}" "${result_dir}"
}

summarize_results() {
  local label="$1"
  local result_dir="$2"
  LABEL="${label}" RESULT_DIR="${result_dir}" python - <<'PY'
import glob
import json
import os

label = os.environ["LABEL"]
result_dir = os.environ["RESULT_DIR"]
files = sorted(glob.glob(os.path.join(result_dir, "diffusion_result_*.json")))
if not files:
    print(f"[Summary] {label}: no result JSON found under {result_dir}")
    raise SystemExit(0)

latest = files[-1]
with open(latest, encoding="utf-8") as f:
    rows = json.load(f)

summary_path = os.path.join(result_dir, f"summary_{label}.tsv")
with open(summary_path, "w", encoding="utf-8") as out:
    out.write("label\tmode\tcase\tthroughput_qps\tlatency_mean\tlatency_p50\tlatency_p99\tpeak_memory_mb_mean\tcompleted\tfailed\n")
    for row in rows:
        test_name = row.get("test_name", "")
        mode = "compile" if test_name.endswith("_compile") else "eager"
        params = row.get("benchmark_params", {})
        result = row.get("result", {})
        out.write(
            "\t".join(
                str(x)
                for x in [
                    label,
                    mode,
                    params.get("name", ""),
                    result.get("throughput_qps", ""),
                    result.get("latency_mean", ""),
                    result.get("latency_p50", ""),
                    result.get("latency_p99", ""),
                    result.get("peak_memory_mb_mean", ""),
                    result.get("completed_requests", result.get("completed", "")),
                    result.get("failed_requests", result.get("failed", "")),
                ]
            )
            + "\n"
        )

print(f"[Summary] {label}: {latest}")
print(f"[Summary] {label}: {summary_path}")
PY
}

echo "Output root: ${OUT_ROOT}"
echo "Model:       ${MODEL}"
echo "Cases:       ${SMALL_WIDTH}x${SMALL_HEIGHT}x${SMALL_FRAMES}f, ${LARGE_WIDTH}x${LARGE_HEIGHT}x${LARGE_FRAMES}f"
echo "Runs/cell:   warmup=${WARMUP_REQUESTS}, measured=${NUM_PROMPTS}, steps=${NUM_INFERENCE_STEPS}"

if [[ "${RUN_BEFORE}" == "1" ]]; then
  run_side before "${BEFORE_REPO}"
fi

if [[ "${RUN_AFTER}" == "1" ]]; then
  run_side after "${AFTER_REPO}"
fi
