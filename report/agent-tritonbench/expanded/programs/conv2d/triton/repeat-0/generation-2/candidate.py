import torch
import triton
import triton.language as tl


@triton.jit
def _conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    total_positions,
    in_channels,
    out_channels,
    in_height,
    in_width,
    out_height,
    out_width,
    input_stride_n,
    input_stride_c,
    input_stride_h,
    input_stride_w,
    weight_stride_o,
    weight_stride_i,
    weight_stride_h,
    weight_stride_w,
    output_stride_n,
    output_stride_c,
    output_stride_h,
    output_stride_w,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DILATION_H: tl.constexpr,
    DILATION_W: tl.constexpr,
    IN_CHANNELS_PER_GROUP: tl.constexpr,
    OUT_CHANNELS_PER_GROUP: tl.constexpr,
    K_TOTAL: tl.constexpr,
    GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    positions = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    output_channels = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    position_mask = positions < total_positions
    channel_mask = output_channels < out_channels

    output_plane_size = out_height * out_width
    batch = positions // output_plane_size
    spatial = positions % output_plane_size
    output_row = spatial // out_width
    output_col = spatial % out_width

    group = output_channels // OUT_CHANNELS_PER_GROUP
    input_channel_base = group * IN_CHANNELS_PER_GROUP

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in tl.static_range(0, K_TOTAL, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K_TOTAL

        input_channel = k_offsets // (KERNEL_H * KERNEL_W)
        kernel_offset = k_offsets % (KERNEL_H * KERNEL_W)
        kernel_row = kernel_offset // KERNEL_W
        kernel_col = kernel_offset % KERNEL_W

        input_row = output_row[:, None] * STRIDE_H - PAD_H + kernel_row[None, :] * DILATION_H
        input_col = output_col[:, None] * STRIDE_W - PAD_W + kernel_col[None, :] * DILATION_W

        input_offsets = (
            batch[:, None] * input_stride_n
            + (input_channel_base[:, None] + input_channel[None, :]) * input_stride_c
            + input_row * input_stride_h
            + input_col * input_stride_w
        )
        input_mask = (
            position_mask[:, None]
            & k_mask[None, :]
            & (input_row >= 0)
            & (input_row < in_height)
            & (input_col >= 0)
            & (input_col < in_width)
        )
        input_tile = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)

        # Load each output-channel row contiguously, then present the dot
        # product with the reduction dimension first.
        weight_offsets = (
            output_channels[:, None] * weight_stride_o
            + input_channel[None, :] * weight_stride_i
            + kernel_row[None, :] * weight_stride_h
            + kernel_col[None, :] * weight_stride_w
        )
        weight_mask = channel_mask[:, None] & k_mask[None, :]
        weight_tile = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0)
        weight_tile = tl.trans(weight_tile)

        accumulator = tl.dot(input_tile, weight_tile, accumulator, input_precision="ieee")

    if HAS_BIAS:
        bias_tile = tl.load(bias_ptr + output_channels, mask=channel_mask, other=0.0)
        accumulator += bias_tile[None, :]

    output_offsets = (
        batch[:, None] * output_stride_n
        + output_channels[None, :] * output_stride_c
        + output_row[:, None] * output_stride_h
        + output_col[:, None] * output_stride_w
    )
    output_mask = position_mask[:, None] & channel_mask[None, :]
    tl.store(output_ptr + output_offsets, accumulator, mask=output_mask)


def _pair(value):
    if isinstance(value, tuple):
        return value
    return value, value


def build(context):
    def wrapper(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        stride_h, stride_w = _pair(stride)
        dilation_h, dilation_w = _pair(dilation)

        if isinstance(padding, str):
            if padding == "valid":
                padding_h, padding_w = 0, 0
            else:
                raise ValueError("only valid padding is supported")
        else:
            padding_h, padding_w = _pair(padding)

        batch_size, input_channels, input_height, input_width = input.shape
        output_channels, channels_per_group, kernel_height, kernel_width = weight.shape
        output_height = (
            (input_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) // stride_h
        ) + 1
        output_width = (
            (input_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) // stride_w
        ) + 1

        output = torch.empty(
            (batch_size, output_channels, output_height, output_width),
            device=input.device,
            dtype=input.dtype,
        )

        input_channels_per_group = input_channels // groups
        output_channels_per_group = output_channels // groups
        total_positions = batch_size * output_height * output_width
        bias_ptr = bias if bias is not None else input

        grid = (
            triton.cdiv(total_positions, 32),
            triton.cdiv(output_channels, 64),
        )
        _conv2d_kernel[grid](
            input,
            weight,
            bias_ptr,
            output,
            total_positions,
            input_channels,
            output_channels,
            input_height,
            input_width,
            output_height,
            output_width,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            input.stride(3),
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            weight.stride(3),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            BLOCK_M=32,
            BLOCK_N=64,
            BLOCK_K=32,
            KERNEL_H=kernel_height,
            KERNEL_W=kernel_width,
            STRIDE_H=stride_h,
            STRIDE_W=stride_w,
            PAD_H=padding_h,
            PAD_W=padding_w,
            DILATION_H=dilation_h,
            DILATION_W=dilation_w,
            IN_CHANNELS_PER_GROUP=input_channels_per_group,
            OUT_CHANNELS_PER_GROUP=output_channels_per_group,
            K_TOTAL=channels_per_group * kernel_height * kernel_width,
            GROUPS=groups,
            HAS_BIAS=bias is not None,
            num_warps=4,
            num_stages=2,
        )
        return output

    return wrapper
