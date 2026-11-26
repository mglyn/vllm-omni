# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact-shape BF16 linear overrides for the eager, unquantized CUDA path."""

import ctypes
import json
import weakref
from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.utils import cuda_unquantized_gemm
from vllm.utils.torch_utils import current_stream

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)
_WORKSPACE_LIMIT = 32 << 20
_ONE = ctypes.c_float(1)
_ZERO = ctypes.c_float(0)


class _TunedGemm:
    def __init__(self, lt, device, table):
        self.lt = lt
        self.device = device
        self.table = table
        self.plans = {}
        self.resources = ExitStack()
        weakref.finalize(self, self.resources.close)
        self.handle = lt.create()
        self.resources.callback(lt.destroy, self.handle)

    def _plan(self, shape, config):
        from nvmath import CudaDataType
        from nvmath.bindings import cublas

        m, n, k = shape
        algo_id, tile, stages, swizzle = config
        lt = self.lt
        f32 = CudaDataType.CUDA_R_32F
        bf16 = CudaDataType.CUDA_R_16BF
        compute = cublas.ComputeType.COMPUTE_32F
        with ExitStack() as stack:
            op = lt.matmul_desc_create(compute, f32)
            stack.callback(lt.matmul_desc_destroy, op)
            transpose = ctypes.c_int(cublas.Operation.T)
            lt.matmul_desc_set_attribute(op, lt.MatmulDescAttribute.TRANSA, ctypes.addressof(transpose), 4)
            # Column-major D^T = W X^T for contiguous row-major X and W.
            layouts = []
            for rows, cols in ((k, n), (k, m), (n, m)):
                layout = lt.matrix_layout_create(bf16, rows, cols, rows)
                stack.callback(lt.matrix_layout_destroy, layout)
                layouts.append(layout)
            a, b, d = layouts
            algo = lt.MatmulAlgo()
            lt.matmul_algo_init(self.handle, compute, f32, bf16, bf16, bf16, bf16, algo_id, algo.ptr)
            attr = lt.MatmulAlgoConfigAttribute
            for field, value in (
                (attr.TILE_ID, tile),
                (attr.STAGES_ID, stages),
                (attr.CTA_SWIZZLING, swizzle),
                (attr.SPLITK_NUM, 1),
            ):
                scalar = ctypes.c_int(value)
                lt.matmul_algo_config_set_attribute(algo.ptr, field, ctypes.addressof(scalar), 4)
            check = lt.MatmulHeuristicResult()
            lt.matmul_algo_check(self.handle, op, a, b, d, d, algo.ptr, check.ptr)
            size = int(check.workspace_size)
            if check.state or size > _WORKSPACE_LIMIT:
                return None
            self.resources.callback(stack.pop_all().close)
            return op, a, b, d, algo, size

    def __call__(self, layer, x, weight, bias):
        if (
            bias is not None
            or x.ndim != 2
            or weight.ndim != 2
            or x.device != self.device
            or weight.device != x.device
            or x.dtype != torch.bfloat16
            or weight.dtype != x.dtype
            or torch.is_grad_enabled()
            or x.is_neg()
            or weight.is_neg()
            or (torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") != x.dtype)
            or not x.is_contiguous()
            or not weight.is_contiguous()
        ):
            return cuda_unquantized_gemm(layer, x, weight, bias)
        shape = (x.size(0), weight.size(0), weight.size(1))
        config = self.table.get(shape)
        if (
            config is None
            or x.size(1) != weight.size(1)
            or x.data_ptr() % 256
            or weight.data_ptr() % 256
            or current_omni_platform.is_graph_capturing()
        ):
            return cuda_unquantized_gemm(layer, x, weight, bias)
        if shape not in self.plans:
            try:
                self.plans[shape] = self._plan(shape, config)
            except self.lt.cuBLASLtError as exc:
                logger.warning_once("Rejected cuBLASLt plan for %s: %s", shape, exc)
                self.plans[shape] = None
        plan = self.plans[shape]
        if plan is None:
            return cuda_unquantized_gemm(layer, x, weight, bias)
        op, a, b, d, algo, size = plan
        output = x.new_empty((shape[0], shape[1]))
        workspace = x.new_empty((size,), dtype=torch.uint8) if size else None
        self.lt.matmul(
            self.handle,
            op,
            ctypes.addressof(_ONE),
            weight.data_ptr(),
            a,
            x.data_ptr(),
            b,
            ctypes.addressof(_ZERO),
            output.data_ptr(),
            d,
            output.data_ptr(),
            d,
            algo.ptr,
            workspace.data_ptr() if workspace is not None else 0,
            size,
            current_stream().cuda_stream,
        )
        return output


def install_tuned_gemm(layers: Iterable[torch.nn.Module], device: torch.device) -> None:
    if not current_omni_platform.is_cuda():
        return
    try:
        from nvmath.bindings import cublasLt as lt  # noqa: N813
    except ImportError:
        logger.warning_once("Tuned BF16 GEMM requires nvmath-python. Using default linear kernels.")
        return
    name = current_omni_platform.get_device_name(device.index or 0).replace(" ", "_").replace("/", "_")
    path = Path(__file__).with_name(f"{name}.bf16.cublaslt-{lt.get_version()}.json")
    if not path.is_file():
        logger.info_once("No tuned BF16 GEMM table matching %s.", path.name)
        return
    raw = json.loads(path.read_text())
    table = {tuple(map(int, key.split(","))): value for key, value in raw.items()}
    dispatch = _TunedGemm(lt, device, table)
    for layer in layers:
        method = layer.quant_method
        # Replace only the default GEMM. The layer retains its TP, loading and
        # batch-invariant logic and any explicitly chosen backend takes precedence.
        if type(method) is UnquantizedLinearMethod and method._gemm_impl is cuda_unquantized_gemm:
            method._gemm_impl = dispatch
