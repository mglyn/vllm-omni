# Diffusion LoRA

This document defines the shared diffusion LoRA loading and execution
contract. For commands and request examples, see the
[Diffusion LoRA user guide](../../user_guide/diffusion/lora.md).

## Ownership and data flow

The common backend owns adapter download, validation, caching, weighted
composition, layer installation, dynamic execution, and startup fusion. A
model owns only the checkpoint-specific knowledge needed to normalize and bind
its adapters:

| Extension | Model describes | Backend performs |
| --- | --- | --- |
| `get_lora_load_plan()` | PEFT metadata, key mapping, tensor conversion, and typed auxiliary updates | Download, deserialize, validate, and cache |
| `get_lora_apply_plan()` | Target components/modules and packed-projection layout | Install layers, activate compositions, and fuse weights |

```text
CLI or request adapters
        -> canonical weighted composition
        -> model load/apply plans
        -> validated A/B tensors and typed updates
        -> dynamic branches or startup-fused dense weights
```

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
module mapping are insufficient. Model code must not duplicate cache,
composition, or fusion lifecycle logic.

## Performance reference

MiniMax-H3 Turbo was measured on 4× H200 with USP4/Ring1, VAE TP4, regional
`torch.compile`, identical 768×1344/107-frame requests, two full-shape warmups,
and five measured runs. Values are median `stage_0_gen_ms`:

| Case | NFE | Stage 0 p50 |
| --- | ---: | ---: |
| Base, no LoRA | 49 | 70.634 s |
| No-LoRA control | 4 | 9.256 s |
| Turbo prefused | 4 | 9.245 s |
| Turbo dynamic | 4 | 9.520 s |

Dynamic Turbo is 7.42× faster than the 49-NFE reference because the Turbo
recipe reduces Transformer evaluations. At the same four NFE, dynamic execution
cost 2.97% relative to prefusion and remained close to the no-LoRA control.
The result supports dynamic LoRA as the serving default; it is not a universal
performance guarantee for other models or adapter ranks.

## Runtime boundaries

- Compile and offload freeze the module graph; preload adapters that cover all
  later target modules and sufficient total rank.
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
