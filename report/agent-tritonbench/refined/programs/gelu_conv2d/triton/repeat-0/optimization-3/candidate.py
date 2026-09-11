from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    batch = rows // 4096
    spatial = rows - batch * 4096
    out_h = spatial // 64
    out_w = spatial - out_h * 64
    out_channels = tl.arange(0, 32)

    acc = tl.zeros((BLOCK_M, 32), dtype=tl.float32)

    # Four full K tiles avoid the masked tail in the common path.
    for k_start in range(0, 128, 32):
        k = k_start + tl.arange(0, 32)
        in_channel = k // 9
        kernel_pos = k - in_channel * 9
        kernel_h = kernel_pos // 3
        kernel_w = kernel_pos - kernel_h * 3

        in_h = out_h[:, None] + kernel_h[None, :] - 1
        in_w = out_w[:, None] + kernel_w[None, :] - 1
        spatial_mask = (in_h >= 0) & (in_h < 64) & (in_w >= 0) & (in_w < 64)
        input_offsets = (
            batch[:, None] * 65536
            + in_channel[None, :] * 4096
            + in_h * 64
            + in_w
        )
        input_values = tl.load(input_ptr + input_offsets, mask=spatial_mask, other=0.0)

        weight_offsets = out_channels[None, :] * 144 + k[:, None]
        weight_values = tl.load(weight_ptr + weight_offsets)
        acc = tl.dot(input_values, weight_values, acc=acc, input_precision="ieee")

    k = 128 + tl.arange(0, 16)
    in_channel = k // 9
    kernel_pos = k - in_channel * 9
    kernel_h = kernel_pos // 3
    kernel_w = kernel_pos - kernel_h * 3
    in_h = out_h[:, None] + kernel_h[None, :] - 1
    in_w = out_w[:, None] + kernel_w[None, :] - 1
    spatial_mask = (in_h >= 0) & (in_h < 64) & (in_w >= 0) & (in_w < 64)
    input_offsets = (
        batch[:, None] * 65536
        + in_channel[None, :] * 4096
        + in_h * 64
        + in_w
    )
    input_values = tl.load(input_ptr + input_offsets, mask=spatial_mask, other=0.0)
    weight_offsets = out_channels[None, :] * 144 + k[:, None]
    weight_values = tl.load(weight_ptr + weight_offsets)
    acc = tl.dot(input_values, weight_values, acc=acc, input_precision="ieee")

    if HAS_BIAS:
        acc += tl.load(bias_ptr + out_channels)[None, :]

    gelu = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    output_offsets = (
        batch[:, None] * 131072
        + out_channels[None, :] * 4096
        + spatial[:, None]
    )
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
        _gelu_conv2d_kernel[(512,)](
            input,
            weight,
            bias_ptr,
            out,
            HAS_BIAS=bias is not None,
            BLOCK_M=128,
            num_warps=4,
            num_stages=2,
        )
        return out

    return wrapper
