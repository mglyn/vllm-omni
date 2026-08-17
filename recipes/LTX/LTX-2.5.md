# LTX-2.5

> Text-to-video and first-frame image-to-video generation with synchronized audio

## Overview

vLLM-Omni supports the gated
[`Lightricks/LTX-2.5-Diffusers`](https://huggingface.co/Lightricks/LTX-2.5-Diffusers)
checkpoint for T2V and first-frame I2V. Every output includes synchronized
48 kHz stereo audio.

The raw [`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5)
repository is not directly loadable with `--model`. It supplies the official
upsampler and LoRA sidecars used by the Full/SFT two-stage pipeline, so accept
both model licenses and authenticate before first use.

## Choose a pipeline

| Pipeline | Mode | Default size | Schedule |
|---|---|---:|---:|
| `LTX2Pipeline` | Full/SFT one-stage | 960x544 | 30 steps |
| `LTX2TwoStagePipeline` | Full/SFT two-stage | 1920x1088 | 30 + 3 steps |
| `LTX2DistilledOneStagePipeline` | Distilled one-stage | 960x544 | 8 steps |
| `LTX2DistilledTwoStagePipeline` | Distilled two-stage | 1920x1088 | 8 + 3 steps |

Both two-stage pipelines generate at half resolution, apply the official x2
latent upsampler, and run a three-step refinement stage. Select the class with
`--model-class-name`; no `--task-type` flag is required. Supplying one initial
image selects I2V, while omitting it selects T2V.

## Diffusion VAE decoder selection

The convolutional VAE decoder remains the default. LTX-2.5 can instead load
the published Diffusion VAE decoder (DiffVAE) by setting the startup-only model
extra `ltx2_use_diffusion_decoder: true` on diffusion stage 0. The choice is
made at engine startup because it changes which weights are loaded; it is not
a per-request sampling parameter. The same extra works with all four public
LTX-2.5 pipeline classes listed above.

For offline Python usage, pass the model extra through the stage override:

```python
omni = Omni(
    model="Lightricks/LTX-2.5-Diffusers",
    model_class_name="LTX2Pipeline",
    stage_overrides='{"0":{"extras":{"ltx2_use_diffusion_decoder":true}}}',
)
```

For online serving, use the same stage override at startup:

```bash
vllm serve Lightricks/LTX-2.5-Diffusers \
  --omni \
  --model-class-name LTX2Pipeline \
  --stage-overrides '{"0":{"extras":{"ltx2_use_diffusion_decoder":true}}}'
```

Both decoders are untiled by default. Choose the decode mode based on memory
and latency:

| Settings | Decode mode | Choose when |
|---|---|---|
| PP=1 without `vae_use_tiling` | One full decode | It fits memory; this is the fastest single-GPU path |
| PP=1 with `vae_use_tiling` | Serial overlapping tiles | Lower peak memory is worth extra tile/blend work |
| `--usp N --vae-patch-parallel-size N` | Tiles distributed over N ranks | Multiple DiT ranks are available and decode latency matters |

For DiffVAE, tiling activates when frames exceed 80 or either spatial dimension
exceeds 768 pixels. Setting `vae_patch_parallel_size>1` enables it automatically;
the option reuses existing DiT ranks rather than launching workers. DiffVAE
stages 1-3 still process the full low-resolution feature volume on every rank,
while stage 4 and the diffusion stage run per tile and rank 0 blends the result.
DiffVAE is decoder-only, so the convolutional VAE remains loaded for I2V encoding.

## Prerequisites

```bash
hf auth login
export MODEL=Lightricks/LTX-2.5-Diffusers
```

Install matching vLLM and vLLM-Omni versions, and ensure `ffmpeg` and
`ffprobe` are on `PATH`. I2V requires PyAV backed by an FFmpeg build with the
`libx264` encoder.

## Hardware

| GPU | Status | Recommended scope |
|---|---|---|
| NVIDIA B300 | Verified | All four canonical pipelines |
| NVIDIA B200 or H200 | Capacity-based recommendation; not yet verified | All four canonical pipelines |
| NVIDIA GB200 or GB300 | Capacity-based recommendation; not yet verified | All four canonical pipelines |
| NVIDIA H100 80 GB | FP8 recipe | Distilled one-stage at 960x544 |

The 1920x1088 two-stage examples require about 114 GB of peak GPU memory.
80 GB GPUs do not have enough safety margin for the canonical two-stage
configuration; use a larger GPU or reduce the output configuration.

### H100 80 GB FP8

Use the distilled one-stage pipeline with FP8 and cuDNN attention:

```bash
vllm serve Lightricks/LTX-2.5-Diffusers \
  --omni \
  --model-class-name LTX2DistilledOneStagePipeline \
  --quantization fp8 \
  --diffusion-attention-backend CUDNN_ATTN \
  --host 0.0.0.0 \
  --port 8000
```

Use the T2V or I2V request below after the server is ready.

## Offline inference

Choose values from the pipeline table. For example, the distilled two-stage
path uses:

```bash
export PIPELINE=LTX2DistilledTwoStagePipeline
export WIDTH=1920
export HEIGHT=1088
export STEPS=8
```

### Text-to-video

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model "${MODEL}" \
  --model-class-name "${PIPELINE}" \
  --prompt "A cinematic shot of a red fox walking through a snowy forest at dawn, the camera tracking alongside, snow crunching underfoot." \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --num-frames 121 \
  --num-inference-steps "${STEPS}" \
  --frame-rate 24 \
  --fps 24 \
  --seed 42 \
  --output ltx25-t2v.mp4
```

### First-frame image-to-video

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model "${MODEL}" \
  --model-class-name "${PIPELINE}" \
  --image /absolute/path/to/first-frame.png \
  --prompt "The red fox walks forward while the camera tracks alongside." \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --num-frames 121 \
  --num-inference-steps "${STEPS}" \
  --frame-rate 24 \
  --fps 24 \
  --seed 42 \
  --output ltx25-i2v.mp4
```

LTX-2.5 uses the official CRF-18 first-frame conditioning path by default.

## Online serving

Start one server for the selected pipeline:

```bash
vllm serve "${MODEL}" \
  --omni \
  --model-class-name "${PIPELINE}" \
  --host 0.0.0.0 \
  --port 8000 \
  --stage-init-timeout 900
```

T2V request:

```bash
curl -sS --fail-with-body \
  -X POST http://127.0.0.1:8000/v1/videos/sync \
  -F 'prompt=A cinematic shot of a red fox walking through a snowy forest at dawn, the camera tracking alongside, snow crunching underfoot.' \
  -o ltx25-online-t2v.mp4
```

For I2V, add exactly one first frame:

```bash
export FIRST_FRAME=/absolute/path/to/first-frame.png

curl -sS --fail-with-body \
  -X POST http://127.0.0.1:8000/v1/videos/sync \
  -F "input_reference=@${FIRST_FRAME};type=image/png" \
  -F 'prompt=The red fox walks forward while the camera tracks alongside.' \
  -o ltx25-online-i2v.mp4
```

Restart the server after changing `PIPELINE` because each class selects a
different weight layout and execution topology.

## Constraints

- Online `width`, `height`, `num_frames`, `fps`, and `num_inference_steps`
  are optional and use the selected pipeline defaults when omitted. `seed` is
  optional and random when omitted.
- `num_frames` must be `8k+1` when overridden.
- One-stage width and height must be divisible by 32; two-stage final
  dimensions must be divisible by 64.
- Full/SFT pipelines accept negative prompts. Distilled pipelines are
  positive-only and reject negative prompts.
- I2V accepts exactly one initial image per prompt.
- The supported `--model` value is `Lightricks/LTX-2.5-Diffusers`.

## References

- [LTX-2.5-Diffusers model card](https://huggingface.co/Lightricks/LTX-2.5-Diffusers)
- [Official LTX-2 repository](https://github.com/Lightricks/LTX-2)
- [Generic T2V example](../../examples/offline_inference/text_to_video/text_to_video.py)
- [Generic I2V example](../../examples/offline_inference/image_to_video/image_to_video.py)
