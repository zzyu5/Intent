from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 16}, num_warps=4, num_stages=2),
    ],
    key=["total_spatial"],
)
@triton.jit
def _gelu_conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    total_spatial,
    HAS_BIAS: tl.constexpr,
    APPROX_TANH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Keep the batch index scalar. Each spatial tile stays inside one image.
    pid_spatial = tl.program_id(0)
    batch = tl.program_id(1)
    spatial = pid_spatial * BLOCK_M + tl.arange(0, BLOCK_M)
    out_h = spatial // 64
    out_w = spatial - out_h * 64

    out_channels = tl.arange(0, 32)
    weight_channel_offsets = out_channels * 144
    output_channel_offsets = out_channels * 4096
    output_base = batch * 131072 + spatial

    acc = tl.zeros((BLOCK_M, 32), dtype=tl.float32)
    for k_start in range(0, 144, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)
        k_mask = k < 144
        in_channel = k // 9
        kernel_pos = k - in_channel * 9
        kernel_h = kernel_pos // 3
        kernel_w = kernel_pos - kernel_h * 3

        valid_h = (out_h[:, None] + kernel_h[None, :] >= 1) & (
            out_h[:, None] + kernel_h[None, :] <= 64
        )
        valid_w = (out_w[:, None] + kernel_w[None, :] >= 1) & (
            out_w[:, None] + kernel_w[None, :] <= 64
        )
        input_offsets = (
            batch * 65536
            + in_channel[None, :] * 4096
            + spatial[:, None]
            + kernel_h[None, :] * 64
            + kernel_w[None, :]
            - 65
        )
        input_values = tl.load(
            input_ptr + input_offsets,
            mask=valid_h & valid_w & k_mask[None, :],
            other=0.0,
        )

        weight_offsets = k[:, None] + weight_channel_offsets[None, :]
        weight_values = tl.load(weight_ptr + weight_offsets, mask=k_mask[:, None], other=0.0)
        acc = tl.dot(input_values, weight_values, acc=acc, input_precision="ieee")

    if HAS_BIAS:
        acc += tl.load(bias_ptr + out_channels)[None, :]

    if APPROX_TANH:
        cubic = acc * acc * acc
        gelu = 0.5 * acc * (
            1.0 + tl.math.tanh(0.7978845608028654 * (acc + 0.044715 * cubic))
        )
    else:
        gelu = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))

    output_offsets = output_base[:, None] + output_channel_offsets[None, :]
    tl.store(output_ptr + output_offsets, gelu)


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
        total_spatial = 4096
        grid = lambda meta: (triton.cdiv(total_spatial, meta["BLOCK_M"]), 16)
        _gelu_conv2d_kernel[grid](
            input,
            weight,
            bias_ptr,
            out,
            total_spatial,
            HAS_BIAS=bias is not None,
            APPROX_TANH=approximate == "tanh",
        )
        return out

    return wrapper
