# LTX implementation

The denoising transformer and diffusion video decoder are owned by Omni. Generic
embeddings, model/config interfaces and other pipeline components can still come
from Diffusers; native ownership does not mean removing the Diffusers dependency.

## Diffusion video decoder

- `ltx2_diffusion_decoder.py`: decoder modules, native checkpoint conversion,
  single-device execution and serial overlapping-tile decode. The ordinary
  decoder is modified from the Apache-2.0 Diffusers implementation at
  `d035dcd7cc7c88e0a154609b62887d50bba9fdc2`. It does not inherit a Diffusers decoder
  or mutate module classes at runtime. Checkpoint parameter names are preserved.
- `ltx2_diffusion_decoder_distributed.py`: tile ownership and distributed decode.
- `ltx2_diffusion_decoder_tiling.py`: adaptive tile geometry and workload planning.
- `ops/diffvae/`: attention and pointwise kernels, independent of model assembly.

The decode API consumes **denormalized** `[B, C, T, H, W]` latents. The pipeline
applies the checkpoint's per-channel statistics before calling it. A fixed RNG
seed, noise schedule, ghost-frame handling and tile order are needed when
comparing outputs because this decoder itself runs a diffusion process.

Stages 1–4 use the required NATTEN Hub backend. The 11x11x11 stage-5 attention uses
Omni TileLang FNA and the NATTEN token-layout helpers. It accepts contiguous CUDA
BF16 tensors with four 64-channel heads and positive batch size. Unsupported
inputs and kernel failures raise; there is no CPU/FlexAttention fallback or
exception-driven switch back to NATTEN for stage 5.

FNA compiles for the active CUDA target rather than a fixed `sm_90a`. H200 is the
currently measured device; other CUDA architectures still need hardware
validation. The existing bit-exact eager pointwise optimizations retain their
separate eligibility checks, including the measured SM90 restriction. Their
unfused operations remain available for other dtypes, shapes and compiled mode.

Source-code licensing and model-weight licensing are separate. This directory's
Apache-2.0 source notices do not grant additional rights to LTX checkpoints;
users obtain model artifacts separately under their applicable model terms.
