# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Install once after weight loading; TP reduction and weight loaders stay on each layer."""

import json
import os
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
from vllm.triton_utils import HAS_TRITON

from vllm_omni.platforms import current_omni_platform

from .profile import OPERATION, Target, load_configs, shape_key

logger = init_logger(__name__)


def make_dispatch(device, target, configs, fallback, trace: Path | None = None):
    """Use the same lookup and eligibility checks for offline confirmation and inference."""
    seen = set()

    def dispatch(layer, x, weight, bias):
        if (
            torch.compiler.is_compiling()
            or torch.is_grad_enabled()
            or bias is not None
            or getattr(layer, "use_skinny_gemm", False)
            or x.ndim < 2
            or weight.ndim != 2
            or x.dtype != torch.bfloat16
            or weight.dtype != x.dtype
            or x.device != device
            or weight.device != x.device
            or not x.is_contiguous()
            or not weight.is_contiguous()
            or not x.numel()
            or not weight.numel()
            or x.size(-1) != weight.size(1)
            or x.is_neg()
            or weight.is_neg()
            or (torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") != x.dtype)
            or current_omni_platform.is_graph_capturing()
        ):
            return fallback(layer, x, weight, bias)
        n, k = weight.shape
        shape = (x.numel() // k, n, k)
        if trace is not None and shape not in seen:
            with trace.open("a") as stream:
                stream.write(
                    json.dumps({"operation": OPERATION, "target": asdict(target), "shape": shape_key(shape)}) + "\n"
                )
            seen.add(shape)
        config = configs.get(shape)
        if config is None:
            return fallback(layer, x, weight, bias)
        if shape not in seen:
            logger.info("Diffusion GEMM matched %s on %s: %s", shape, target.device, config)
            seen.add(shape)
        return config.run(x, weight)

    return dispatch


def install(pipeline: torch.nn.Module, device: torch.device, options: dict) -> None:
    if not isinstance(options, dict):
        raise ValueError("gemm_tuning must be an object")
    mode = options.get("mode")
    if mode not in ("collect", "dispatch"):
        raise ValueError("gemm_tuning mode must be 'collect' or 'dispatch'")
    location = options.get("directory")
    if location is None and mode == "dispatch":
        directory = Path(__file__).with_name("configs")
    elif isinstance(location, str) and location:
        directory = Path(location)
    else:
        raise ValueError("gemm_tuning requires a nonempty directory")
    device = torch.device(device)
    if (
        device.type != "cuda"
        or not HAS_TRITON
        or not (current_omni_platform.is_cuda() or current_omni_platform.is_rocm())
    ):
        logger.info("No diffusion GEMM tuning backend for %s", device)
        return
    if device.index is None:
        device = torch.device(device.type, torch.accelerator.current_device_index())
    target = Target.current(device)
    configs = load_configs(directory, target) if mode == "dispatch" else {}
    if mode == "dispatch" and not configs:
        logger.info("No optimized entries for %s; keeping the default linear implementation", target.filename)
        return
    trace = None
    if mode == "collect":
        directory.mkdir(parents=True, exist_ok=True)
        trace = directory / f"shapes-{os.getpid()}-{uuid4().hex}.jsonl"
    fallback = dispatch_unquantized_gemm()
    dispatch = make_dispatch(device, target, configs, fallback, trace)
    weight_shapes = {(n, k) for _, n, k in configs}
    installed = 0
    for layer in pipeline.modules():
        method = getattr(layer, "quant_method", None)
        if (
            method is None
            or type(method) is not UnquantizedLinearMethod
            or getattr(method, "_gemm_impl", None) is not fallback
        ):
            continue
        weight = getattr(layer, "weight", None)
        if mode == "dispatch" and weight is not None and weight.ndim == 2 and tuple(weight.shape) not in weight_shapes:
            continue
        method._gemm_impl = dispatch
        installed += 1
    logger.info(
        "Diffusion GEMM %s: %d eligible linear modules, target=%s, file=%s",
        mode,
        installed,
        target.device,
        trace or directory / target.filename,
    )
