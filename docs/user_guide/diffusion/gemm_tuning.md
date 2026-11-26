# Device-specific diffusion GEMM dispatch

The diffusion worker can load measured configurations for eager BF16 linear layers with contiguous activations, contiguous transposed weights and no bias. Enable dispatch with the following command.

```bash
vllm serve "$LOCAL_MODEL" --omni --enforce-eager \
    --additional-config '{"gemm_tuning":{"mode":"dispatch","directory":"/absolute/path/to/gemm-tables"}}'
```

Omit `directory` in dispatch mode to load the profiles distributed with vLLM-Omni. A custom directory takes precedence. To contribute a device profile, place its JSON file under `vllm_omni/diffusion/layers/gemm_tuning/configs/` and include the workload, raw tuning report and end-to-end comparison in the PR.

The worker looks up a versioned JSON table by backend, GPU name, architecture, compute-unit count, PyTorch, vLLM, Triton and CUDA/ROCm build versions. Each entry uses an exact `M,N,K` key. A table can select the default backend, a Triton tile, or padding of the flattened activation to a measured row alignment followed by the native linear and a slice back to the logical output shape. Padding cost is included in timing. No interpolation between measured shapes takes place. Two neighboring resolutions can reuse an entry when their physical GEMM dimensions match.

Missing tables, default entries and unseen shapes use the existing vLLM implementation. Bias, gradients, incompatible layouts, specialized linear methods, compiled code and graph capture also take their existing paths. Workers report the first use of each optimized shape. A matching table with malformed entries raises a configuration error at load time.

## Validate a device profile

Compare warmed end-to-end requests on the same device with dispatch disabled and enabled. Keep prompt, seed, internal canvas, batch, steps, offload and attention settings fixed. Record image output and complete request latency. For H3-Image, `use_tuned_bf16_gemm` controls a separate, model-owned packed-sequence optimization. Keep that switch fixed when measuring kernel dispatch. CUDA A100 is the first tested target. Results on other accelerators require measurements on those devices.

For H3 packed-row experiments on a new CUDA SKU, the server accepts an explicit table, for example `{"use_tuned_bf16_gemm":true,"bf16_gemm_row_profiles":{"<exact device name>":{"4112":4096}}}`. These numbers illustrate the format. Each experiment must preserve all valid rows and a separate nonempty padding document. Benchmark the candidate against the release row count at the same canvas and compare decoded RGB.
