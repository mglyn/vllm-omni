# Tune diffusion GEMMs for a GPU

The diffusion worker can collect the local matrix dimensions of unquantized vLLM linear layers and load an offline tuning table. Models using these layers participate automatically. The first operation is a contiguous BF16 activation times a contiguous transposed BF16 weight, with no bias. The runner leaves tuning disabled by default.

## Benchmark a new GPU

Run this command from the checkout on the target machine with its usual vLLM-Omni environment. The supplied workload is the set of 20 distinct operand shapes observed during server startup and one six-step MiniMax-H3-Image 1920×1080 request on a single A100. Replaying it only requires device memory for one synthetic GEMM at a time.

```bash
python benchmarks/kernels/tune_diffusion_gemm.py \
    --trace benchmarks/kernels/gemm_shapes/minimax_h3_image_1080p.jsonl \
    --output-dir ./gemm-tables --report ./measurements.json
```

Use a fresh report path for each run. Use repeated `--shape M,N,K` arguments for individual projections, or pass several trace files after `--trace`. The tuning process detects the target locally. Run the same command on a second device to produce its own table. Serve and tune with the same vLLM and math settings.

## Collect a different workload

Start your usual local diffusion service with shape collection enabled. Use absolute directories accessible to each participating worker.

```bash
vllm serve "$LOCAL_MODEL" --omni --enforce-eager \
    --additional-config '{"gemm_tuning":{"mode":"collect","directory":"/tmp/gemm-traces"}}'
```

Send representative requests, including the resolutions, batches, prompt lengths and parallelism configurations you want to optimize. Each worker writes one JSONL record per distinct eligible shape. For a local weight of shape `[N,K]`, `M` is the product of the leading activation dimensions. The trace contains shape and platform metadata. Re-run the benchmark with `--trace /tmp/gemm-traces/*.jsonl`. The tuner unions the shape sets and benchmarks on the current device.

## Load the table

```bash
vllm serve "$LOCAL_MODEL" --omni --enforce-eager \
    --additional-config '{"gemm_tuning":{"mode":"dispatch","directory":"/absolute/path/to/gemm-tables"}}'
```

Omit `directory` in dispatch mode to load the profiles distributed with vLLM-Omni. A custom directory takes precedence. To contribute a device profile, place its generated JSON file under `vllm_omni/diffusion/layers/gemm_tuning/configs/` and include the workload, raw tuning report and end-to-end comparison in the PR.

The worker looks up a versioned JSON table by backend, GPU name, architecture, compute-unit count, PyTorch, vLLM, Triton and CUDA/ROCm build versions. Each entry uses an exact `M,N,K` key. A table can select the default backend, a Triton tile, or padding of the flattened activation to a measured row alignment followed by the native linear and a slice back to the logical output shape. Padding cost is included in timing. No interpolation between measured shapes takes place. Two neighboring resolutions can reuse an entry when their physical GEMM dimensions match.

Missing tables, default entries and unseen shapes use the existing vLLM implementation. Bias, gradients, incompatible layouts, specialized linear methods, compiled code and graph capture also take their existing paths. Workers report the first use of each optimized shape. A matching table with malformed entries raises a configuration error at load time.

## Interpret the results

The tuner compares against the local vLLM default. It screens tile sizes, warp and stage counts, and distinct native row alignments. It checks BF16 output equality on random operands, repeats that check on new operands for the fastest candidate, and alternates candidate/default timing order. The confirmation includes the same shape lookup and eligibility checks used by the inference worker. The default acceptance threshold is a median latency reduction of at least 2% and a win in the majority of the five repeated pairs. Raw candidate measurements and decisions go in the report. A table containing only default entries is a valid result.

Compare warmed end-to-end requests on the same device with dispatch disabled and enabled. Keep prompt, seed, internal canvas, batch, steps, offload and attention settings fixed. Record image output and complete request latency. For H3-Image, `use_tuned_bf16_gemm` controls a separate, model-owned packed-sequence optimization. Keep that switch fixed when measuring kernel dispatch. CUDA A100 is the first tested target. Results on other accelerators require measurements on those devices.

For H3 packed-row experiments on a new CUDA SKU, the server accepts an explicit table, for example `{"use_tuned_bf16_gemm":true,"bf16_gemm_row_profiles":{"<exact device name>":{"4112":4096}}}`. These numbers illustrate the format. Each experiment must preserve all valid rows and a separate nonempty padding document. Benchmark the candidate against the release row count at the same canvas and compare decoded RGB.

## Add a backend

Implement a configuration dataclass in `vllm_omni/diffusion/layers/gemm_tuning/backends.py` with a stable backend name, validated parameters, `candidates(target, shape)` for offline enumeration and `run(x, weight)` for execution. Add its type to `GemmConfig` and register it in `BACKENDS`. The tuner and worker use the same implementation. Each winning row stores its backend name and configuration fields. A new device can supply a profile through the existing registry without changing model code.
