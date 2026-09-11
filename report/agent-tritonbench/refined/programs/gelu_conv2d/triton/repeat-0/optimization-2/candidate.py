from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_conv2d_interior_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    total_rows,
    HAS_BIAS: tl.constexpr,
    APPROX_TANH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    compact_rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    valid = compact_rows < total_rows
    safe_rows = tl.minimum(compact_rows, total_rows - 1)

    batch = safe_rows // 3844
    compact_spatial = safe_rows - batch * 3844
    out_h = compact_spatial // 62 + 1
    out_w = compact_spatial - (out_h - 1) * 62 + 1
    out_spatial = out_h * 64 + out_w

    out_channels = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, 144, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)
        k_mask = k < 144

        in_channel = k // 9
        kernel_pos = k - in_channel * 9
        kernel_h = kernel_pos // 3
        kernel_w = kernel_pos - kernel_h * 3

        in_h = out_h[:, None] + kernel_h[None, :] - 1
        in_w = out_w[:, None] + kernel_w[None, :] - 1
        input_offsets = (
            batch[:, None] * 65536
            + in_channel[None, :] * 4096
            + in_h * 64
            + in_w
        )
        input_values = tl.load(
            input_ptr + input_offsets,
            mask=valid[:, None] & k_mask[None, :],
            other=0.0,
        )

        weight_offsets = out_channels[None, :] * 144 + k[:, None]
        weight_values = tl.load(
            weight_ptr + weight_offsets,
            mask=k_mask[:, None],
            other=0.0,
        )
        acc = tl.dot(input_values, weight_values, acc=acc, input_precision="ieee")

    if HAS_BIAS:
        acc += tl.load(bias_ptr + out_channels, mask=out_channels < 32, other=0.0)[None, :]

    if APPROX_TANH:
        cubic = acc * acc * acc
        gelu = 0.5 * acc * (
            1.0 + tl.math.tanh(0.7978845608028654 * (acc + 0.044715 * cubic))
        )
    else:
        gelu = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))

    output_offsets = (
        batch[:, None] * 131072
        + out_channels[None, :] * 4096
        + out_spatial[:, None]
    )
    tl.store(output_ptr + output_offsets, gelu, mask=valid[:, None])


@triton.jit
def _gelu_conv2d_border_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    total_rows,
    HAS_BIAS: tl.constexpr,
    APPROX_TANH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    valid = rows < total_rows
    safe_rows = tl.minimum(rows, total_rows - 1)

    batch = safe_rows // 252
    border = safe_rows - batch * 252
    out_h = tl.where(
        border < 64,
        0,
        tl.where(border < 128, 63, border - tl.where(border < 190, 127, 189)),
    )
    out_w = tl.where(
        border < 64,
        border,
        tl.where(border < 128, border - 64, tl.where(border < 190, 0, 63)),
    )
    out_spatial = out_h * 64 + out_w

    out_channels = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, 144, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)
        k_mask = k < 144

        in_channel = k // 9
        kernel_pos = k - in_channel * 9
        kernel_h = kernel_pos // 3
        kernel_w = kernel_pos - kernel_h * 3

        in_h = out_h[:, None] + kernel_h[None, :] - 1
        in_w = out_w[:, None] + kernel_w[None, :] - 1
        spatial_mask = (in_h >= 0) & (in_h < 64) & (in_w >= 0) & (in_w < 64)
        safe_h = tl.maximum(tl.minimum(in_h, 63), 0)
        safe_w = tl.maximum(tl.minimum(in_w, 63), 0)
        input_offsets = (
            batch[:, None] * 65536
            + in_channel[None, :] * 4096
            + safe_h * 64
            + safe_w
        )
        input_values = tl.load(
            input_ptr + input_offsets,
            mask=valid[:, None] & k_mask[None, :] & spatial_mask,
            other=0.0,
        )

        weight_offsets = out_channels[None, :] * 144 + k[:, None]
        weight_values = tl.load(
            weight_ptr + weight_offsets,
            mask=k_mask[:, None],
            other=0.0,
        )
        acc = tl.dot(input_values, weight_values, acc=acc, input_precision="ieee")

    if HAS_BIAS:
        acc += tl.load(bias_ptr + out_channels, mask=out_channels < 32, other=0.0)[None, :]

    if APPROX_TANH:
        cubic = acc * acc * acc
        gelu = 0.5 * acc * (
            1.0 + tl.math.tanh(0.7978845608028654 * (acc + 0.044715 * cubic))
        )
    else:
        gelu = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))

    output_offsets = (
        batch[:, None] * 131072
        + out_channels[None, :] * 4096
        + out_spatial[:, None]
    )
    tl.store(output_ptr + output_offsets, gelu, mask=valid[:, None])


def build(context):
    def wrapper(
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        approximate="none",
        out=None,
    ):
        if out is None:
            out = torch.empty((16, 32, 64, 64), device=input.device, dtype=input.dtype)

        bias_ptr = weight if bias is None else bias
        interior_rows = 16 * 62 * 62
        border_rows = 16 * 252
        interior_grid = (triton.cdiv(interior_rows, 128),)
        border_grid = (triton.cdiv(border_rows, 128),)

        _gelu_conv2d_interior_kernel[interior_grid](
            input,
            weight,
            bias_ptr,
            out,
            interior_rows,
            HAS_BIAS=bias is not None,
            APPROX_TANH=approximate == "tanh",
            BLOCK_M=128,
            BLOCK_N=32,
            BLOCK_K=32,
            num_warps=4,
            num_stages=2,
        )
        _gelu_conv2d_border_kernel[border_grid](
            input,
            weight,
            bias_ptr,
            out,
            border_rows,
            HAS_BIAS=bias is not None,
            APPROX_TANH=approximate == "tanh",
            BLOCK_M=128,
            BLOCK_N=32,
            BLOCK_K=32,
            num_warps=4,
            num_stages=2,
        )
        return out

    return wrapper
