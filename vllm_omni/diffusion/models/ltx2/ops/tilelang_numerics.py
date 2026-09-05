# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Shared exact-numerics constants for LTX-2 TileLang kernels."""

RMS_THREADS = 128
RMS_RSQRT_PRELUDE = r"""
__device__ __forceinline__ float ltx_rsqrt_approx_f32(float value) {
  float result;
  asm("rsqrt.approx.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}
"""

__all__ = ["RMS_RSQRT_PRELUDE", "RMS_THREADS"]
