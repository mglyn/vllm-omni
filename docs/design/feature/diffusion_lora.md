# Diffusion LoRA architecture

This document defines the shared diffusion LoRA loading and execution
contract. For commands and request examples, see the
[Diffusion LoRA user guide](../../user_guide/diffusion/lora.md).

## Ownership and data flow

The common backend owns startup adapter loading, immutable registration,
weighted composition, layer installation, dynamic execution, and prefusion.
A model owns only the checkpoint-specific knowledge needed to
normalize and bind its adapters:

| Extension | Model describes | Backend performs |
| --- | --- | --- |
| `get_lora_load_plan()` | PEFT metadata, key mapping, tensor conversion, and typed auxiliary updates | Load, deserialize, validate, and register at startup |
| `get_lora_apply_plan()` | Target components/modules and packed-projection layout | Install layers, activate compositions, and fuse weights |

```text
startup prefusion / dynamic registration
        -> model load/apply plans
        -> validated A/B tensors and typed updates
        -> fused dense weights / immutable dynamic registry
        -> request-selected weighted composition
```

Requests select dynamic adapters by their unique deployment name. Admission
resolves that name to the canonical server-owned adapter record before
scheduling; request payloads neither contain server paths nor invoke the loader.

Sampling remains outside the LoRA backend. Acceleration adapters may require a
particular scheduler, guidance setting, or number of steps, but deployments and
requests select those parameters independently.

## Execution modes

For adapters with weight deltas $B_iA_i$ and scales $s_i$, both modes implement
the same additive update in exact arithmetic:

$$
y = Wx + b + \sum_i s_i B_i A_i x
$$

- **Dynamic** keeps the base weight unchanged and evaluates the low-rank branch
  during each forward. Requests can select, disable, reweight, and compose
  adapters. A quantized base remains supported.
- **Prefused** materializes the weighted deltas into dense weights once during
  startup. It is permanent and is rejected for quantized weights.

Both modes use the same canonical ordered composition and packed-projection
mapping. BF16 rounding order can still make their generated outputs differ.

## Reference model integrations

The following integrations demonstrate that publication formats remain
model-owned while execution is shared:

| Model | Published adapter example | Model-owned handling | Dynamic / prefused |
| --- | --- | --- | --- |
| MiniMax-H3 | `lightx2v/Minimax-h3-Turbo` | Maps PEFT-like keys, folds the published `alpha / rank`, and targets FL2VA only | Both |
| Qwen-Image | `lightx2v/Qwen-Image-2512-Lightning` | Converts published keys and alpha values, then binds packed QKV projections | Both |
| Wan2.1/2.2 | `lightx2v/Wan2.1-Distill-Loras`, `lightx2v/Wan2.2-Distill-Loras` | Routes high/low-noise adapters and converts supported bias updates into typed updates | Both |

New models should implement plans only when the generic PEFT path and inferred
module mapping are insufficient. Model code must not duplicate registration,
composition, or fusion logic.

## Runtime boundaries

- Every request-selectable adapter is registered with `--dynamic-lora` before
  serving. Request processing selects and reweights registered tensors; it
  never invokes the loader or downloads from the Hub.
- The diffusion registry is immutable while serving; runtime add, remove, and
  pin operations are rejected.
- Compile and offload freeze the module graph after startup registration.
- DLO supports startup dynamic LoRA through ordinary CPU loading; A/B slots
  remain device-resident while DLO streams the base weights. Prefused LoRA with
  DLO is rejected.
- Requests with different canonical compositions use different diffusion batch
  keys.
- When diffusion prefix K/V becomes publishable across requests, its block-hash
  identity must also include the canonical LoRA composition. Current requests
  without canonical block hashes remain request-local and are not published.
- Prefused and dynamic adapters may coexist; specifying the same adapter in
  both sets intentionally applies it twice.
- Unknown typed updates and unsupported nonzero dense deltas are rejected.
