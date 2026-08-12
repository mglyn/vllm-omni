# Diffusion LoRA

vLLM-Omni provides a shared LoRA backend for diffusion pipelines. It supports
startup fusion, dynamic execution, request-scoped adapter selection, weighted
multi-adapter composition, and common adapter lifecycle management.

For a linear layer with base weight $W$, adapter matrices $A_i$ and $B_i$, and
user scales $s_i$, both execution modes implement:

$$
y = \left(W + \sum_i s_i B_i A_i\right)x
$$

The modes differ in when this expression is evaluated:

| Mode | Startup option | Weight lifecycle | Request switching | Quantized base weights |
|---|---|---|---|---|
| Dynamic | `--dynamic-lora PATH=SCALE` | Keeps $W$ unchanged and evaluates the low-rank branch during each forward | Yes | Yes |
| Prefused | `--prefused-lora PATH=SCALE` | Merges the delta into dense weights once at startup | No | No |

Dynamic LoRA is the recommended default for serving. Prefusion is useful only
when permanent dense weights are required and its output quality has been
validated for the model and dtype.

## Adapter formats

The generic loader accepts a PEFT directory containing `adapter_config.json`
and adapter weights:

```text
lora_adapter/
├── adapter_config.json
└── adapter_model.safetensors
```

The PEFT configuration describes the rank, alpha, and target modules. A local
path or Hugging Face repository ID may be used at startup or in a request.
Paths in API requests are resolved by the server, not uploaded by the client.

Single-file `.safetensors` adapters are also supported when their checkpoint
layout is compatible with the selected pipeline. Check the corresponding
recipe for supported adapter repositories and formats.

Model-owned single-file adapters currently include MiniMax-H3 Turbo,
Qwen-Image, Wan2.1 T2V, and Wan2.2 T2V. Wan2.2 assigns adapters containing
`high_noise` and `low_noise` in their filenames to the corresponding
transformers, so the two files can be passed in either CLI order. Current
Wan2.2 I2V LightX2V files also contain dense and bias deltas; use the offline
assembly workflow below because those tensors are not low-rank LoRA terms.

## Dynamic serving

Install one adapter at startup and make it the deployment default:

```bash
vllm serve BASE_MODEL \
  --omni \
  --dynamic-lora /path/to/adapter=1.0 \
  --max-cpu-loras 1
```

Repeat `--dynamic-lora` to install a weighted default composition. The cache
must have room for every adapter that must remain resident:

```bash
vllm serve BASE_MODEL \
  --omni \
  --dynamic-lora /path/to/accelerator.safetensors=1.0 \
  --dynamic-lora org/style-adapter=0.6 \
  --max-cpu-loras 2
```

Startup specifications accept `PATH` or `PATH=SCALE`. When a path itself ends
in text that cannot be parsed as a number, the whole value remains the path.

### Request selection

The Images and Videos APIs accept a `lora` object or list. Each entry requires
`name` and `path`; `scale` defaults to `1.0`, and `int_id` is optional:

```json
{
  "lora": [
    {"name": "accelerator", "path": "/path/to/accelerator.safetensors", "scale": 1.0},
    {"name": "style", "path": "org/style-adapter", "scale": 0.6}
  ]
}
```

Request behavior is explicit:

| Request `lora` value | Dynamic adapters used by the request |
|---|---|
| Field omitted or `null` | Startup `--dynamic-lora` composition |
| `[]` | None |
| One object | That adapter only |
| List of objects | That weighted composition; it replaces the startup default |

For example, a synchronous video request can reweight the startup adapter:

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/videos/sync \
  -F 'model=BASE_MODEL' \
  -F 'prompt=A cinematic wide shot of a singer on an open-air stage.' \
  -F 'lora={"name":"accelerator","path":"/path/to/accelerator.safetensors","scale":0.8}'
```

The same object or list can be passed as the `lora` field of an Images API
JSON request. Adapter names are labels; the path-derived stable integer ID is
the default cache identity. If `int_id` is supplied explicitly, do not reuse
that ID for a different path.

Duplicate adapter IDs in one composition have their scales added, zero-scale
results are removed, and non-finite scales are rejected. Requests with
different adapter compositions are scheduled in separate diffusion batches.

## Prefusion

Use `--prefused-lora` to merge one or more weighted adapters into the dense
weights at startup:

```bash
vllm serve BASE_MODEL \
  --omni \
  --prefused-lora /path/to/accelerator.safetensors=1.0 \
  --prefused-lora /path/to/style-adapter=0.6
```

The backend accumulates each dense delta in FP32 and copies the merged result
to the base dtype once. The fused contribution is permanent for the lifetime
of the process: request-level `lora=[]` disables only dynamic adapters and
cannot remove a prefused delta.

Prefused and dynamic adapters may be used together. The resulting weight is:

$$
W' = W + \sum_{i \in P} s_i B_i A_i
$$

and a request evaluates:

$$
y = W'x + \sum_{j \in D} s_j B_j A_j x
$$

Do not specify the same adapter in both sets unless applying its delta twice
is intentional. Prefusion is rejected for quantized diffusion weights because
their serialized or runtime representation is not a dense floating-point
weight that can safely receive an in-place LoRA delta.

## Compile, offload, and cache capacity

Dynamic adapters are installed before compilation, CPU/layerwise offload, and
diffusion cache wrapping. These features fix the module graph, so preload an
adapter that covers every target layer and enough total rank for later request
compositions. After the graph is fixed, requests may select, disable, or
reweight installed-compatible adapters, but they cannot introduce a new
target layer or expand the allocated rank.

`--max-cpu-loras` controls the per-worker adapter cache. Startup dynamic
adapters are pinned, while request-loaded adapters use LRU eviction. Set the
value to at least the number of pinned startup adapters plus the largest
request-only composition that must coexist with them.

Dynamic LoRA executes the dense base layer through its configured quantization
method and adds the low-rank branch separately, so it can be combined with a
quantized base model. This does not imply bitwise equality with an unquantized
reference.

## Sampling remains model-owned

Loading an acceleration or distilled LoRA does not change timesteps, guidance,
the scheduler, or the number of denoising steps. Set those independently
through deployment defaults or request sampling parameters according to the
adapter's published usage instructions.

## Offline inference

The same request types are available through the Python API:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest

adapter_path = "/path/to/lora_adapter"
omni = Omni(
    model="stabilityai/stable-diffusion-3.5-medium",
    dynamic_lora=[f"{adapter_path}=1.0"],
)

params = OmniDiffusionSamplingParams(
    num_inference_steps=28,
    lora_request=LoRARequest(
        lora_name="style",
        lora_int_id=1,
        lora_path=adapter_path,
    ),
    lora_scale=0.8,
)
outputs = omni.generate("A piece of cheesecake", params)
```

For multiple adapters, pass matching tuples of `LoRARequest` and scales.

## Wan2.2 LightX2V Offline Assembly

This workflow is LoRA-adjacent: it uses external LightX2V conversion plus
`Wan2.2-Distill-Loras` to bake converted Wan2.2 I2V checkpoints into a local
Diffusers directory, instead of loading LoRA adapters at runtime.

### Required assets

- Base model: `Wan-AI/Wan2.2-I2V-A14B`
- Diffusers skeleton: `Wan-AI/Wan2.2-I2V-A14B-Diffusers`
- Optional external converter from the LightX2V project (not shipped in this repository)
- Optional LoRA weights: `lightx2v/Wan2.2-Distill-Loras`

### Step 1: Optional - convert high/low-noise DiT weights with LightX2V

Install or clone LightX2V from the upstream repository
(`https://github.com/ModelTC/LightX2V`). After cloning, the converter used
below is available at `<lightx2v_root>/tools/convert/converter.py`.

```bash
python /path/to/lightx2v/tools/convert/converter.py \
  --source /path/to/Wan2.2-I2V-A14B/high_noise_model \
  --output /tmp/wan22_lightx2v/high_noise_out \
  --output_ext .safetensors \
  --output_name diffusion_pytorch_model \
  --model_type wan_dit \
  --direction forward \
  --lora_path /path/to/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors \
  --lora_key_convert auto \
  --single_file

python /path/to/lightx2v/tools/convert/converter.py \
  --source /path/to/Wan2.2-I2V-A14B/low_noise_model \
  --output /tmp/wan22_lightx2v/low_noise_out \
  --output_ext .safetensors \
  --output_name diffusion_pytorch_model \
  --model_type wan_dit \
  --direction forward \
  --lora_path /path/to/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors \
  --lora_key_convert auto \
  --single_file
```

If you are not using LightX2V, skip this step and either keep the original
Diffusers weights from the skeleton or point Step 2 at any other converted
`transformer/` and `transformer_2/` checkpoints.

### Step 2: Assemble a final Diffusers-style directory

```bash
python tools/wan22/assemble_wan22_i2v_diffusers.py \
  --diffusers-skeleton /path/to/Wan2.2-I2V-A14B-Diffusers \
  --transformer-weight /tmp/wan22_lightx2v/high_noise_out \
  --transformer-2-weight /tmp/wan22_lightx2v/low_noise_out \
  --output-dir /path/to/Wan2.2-I2V-A14B-Custom-Diffusers \
  --asset-mode symlink \
  --overwrite
```

`--transformer-weight` and `--transformer-2-weight` are optional. If you omit
them, the tool keeps the original weights from the Diffusers skeleton.

### Step 3: Run offline inference

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model /path/to/Wan2.2-I2V-A14B-Custom-Diffusers \
  --image /path/to/input.jpg \
  --prompt "A cat playing with yarn" \
  --num-frames 81 \
  --num-inference-steps 4 \
  --tensor-parallel-size 4 \
  --height 480 \
  --width 832 \
  --flow-shift 12 \
  --sample-solver euler \
  --guidance-scale 1.0 \
  --guidance-scale-high 1.0 \
  --boundary-ratio 0.875
```

Notes:

- This route avoids runtime LoRA loading changes in vLLM-Omni when you choose to bake converted weights into a local Diffusers directory.
- Output quality and speed depend on the replacement checkpoints and sampling params you choose.


## See Also

- [Text-to-Image Offline Example](../examples/offline_inference/text_to_image.md#lora) - Complete offline LoRA example
- [Text-to-Image Online Example](../examples/online_serving/text_to_image.md#lora) - Complete online LoRA example
