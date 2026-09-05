# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""TileLang kernel for exact DiffVAE residual, RMSNorm, and AdaLN."""

import torch
from vllm.tilelang_utils import T, tilelang

from ..tilelang_numerics import RMS_RSQRT_PRELUDE

_HIDDEN_SIZE = 256
_RMS_THREADS = 64


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def _residual_rms_norm_modulate_tilelang_kernel(
    residual_count: int,
    eps: float,
):
    rows = T.dynamic("rows")
    batch = T.dynamic("batch")

    @T.prim_func
    def main(
        x: T.Tensor((rows, _HIDDEN_SIZE), T.bfloat16),
        residual_a: T.Tensor((rows, _HIDDEN_SIZE), T.bfloat16),
        residual_b: T.Tensor((rows, _HIDDEN_SIZE), T.bfloat16),
        norm_weight: T.Tensor((_HIDDEN_SIZE,), T.bfloat16),
        scale: T.Tensor((batch, _HIDDEN_SIZE), T.bfloat16),
        shift: T.Tensor((batch, _HIDDEN_SIZE), T.bfloat16),
        output: T.Tensor((rows, _HIDDEN_SIZE), T.bfloat16),
    ):
        with T.Kernel(
            rows,
            threads=_RMS_THREADS,
            prelude=RMS_RSQRT_PRELUDE,
        ) as row:
            tx = T.get_thread_binding(0)
            rows_per_batch = rows // batch
            batch_index = row // rows_per_batch
            values = T.alloc_local((4,), T.float32)
            accumulator = T.alloc_var(T.float32, 0.0)
            warp_sums = T.alloc_shared((2,), T.float32)

            column_base = tx * 4
            for lane in T.unroll(4):
                column = column_base + lane
                value = (
                    T.ieee_add(
                        x[row, column].astype(T.float32),
                        residual_a[row, column].astype(T.float32),
                    )
                    .astype(T.bfloat16)
                    .astype(T.float32)
                )
                if residual_count == 2:
                    value = (
                        T.ieee_add(
                            value,
                            residual_b[row, column].astype(T.float32),
                        )
                        .astype(T.bfloat16)
                        .astype(T.float32)
                    )
                values[lane] = value
                accumulator = T.ieee_fmaf(value, value, accumulator)

            accumulator = T.ieee_add(accumulator, T.shfl_down(accumulator, 16))
            accumulator = T.ieee_add(accumulator, T.shfl_down(accumulator, 8))
            accumulator = T.ieee_add(accumulator, T.shfl_down(accumulator, 4))
            accumulator = T.ieee_add(accumulator, T.shfl_down(accumulator, 2))
            accumulator = T.ieee_add(accumulator, T.shfl_down(accumulator, 1))
            if tx % 32 == 0:
                warp_sums[tx // 32] = accumulator
            T.sync_threads()
            total = T.ieee_add(warp_sums[0], warp_sums[1])
            reciprocal_rms = T.call_extern(
                "float32",
                "ltx_rsqrt_approx_f32",
                T.ieee_add(T.ieee_mul(total, 1.0 / _HIDDEN_SIZE), eps),
            )

            for lane in T.vectorized(4):
                column = column_base + lane
                normalized = T.ieee_mul(values[lane], reciprocal_rms)
                normalized_weighted = (
                    T.ieee_mul(
                        normalized,
                        norm_weight[column].astype(T.float32),
                    )
                    .astype(T.bfloat16)
                    .astype(T.float32)
                )
                scale_value = scale[batch_index, column].astype(T.float32)
                shift_value = shift[batch_index, column].astype(T.float32)
                one_plus_scale = T.ieee_add(1.0, scale_value).astype(T.bfloat16).astype(T.float32)
                product = T.ieee_mul(normalized_weighted, one_plus_scale).astype(T.bfloat16).astype(T.float32)
                output[row, column] = T.ieee_add(product, shift_value)

    return main


def launch_residual_rms_norm_modulate_tilelang(
    x: torch.Tensor,
    residual_a: torch.Tensor,
    residual_b: torch.Tensor | None,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Launch exact TileLang residual addition, RMSNorm, and AdaLN."""

    output = torch.empty_like(x)
    x_rows = x.view(-1, _HIDDEN_SIZE)
    residual_b_arg = x_rows if residual_b is None else residual_b.view(-1, _HIDDEN_SIZE)
    kernel = _residual_rms_norm_modulate_tilelang_kernel(
        residual_count=1 if residual_b is None else 2,
        eps=float(eps),
    )
    with torch.accelerator.device_index(x.device.index):
        kernel(
            x_rows,
            residual_a.view(-1, _HIDDEN_SIZE),
            residual_b_arg,
            norm_weight,
            scale.view(x.shape[0], _HIDDEN_SIZE),
            shift.view(x.shape[0], _HIDDEN_SIZE),
            output.view(-1, _HIDDEN_SIZE),
        )
    return output


__all__ = ["launch_residual_rms_norm_modulate_tilelang"]
