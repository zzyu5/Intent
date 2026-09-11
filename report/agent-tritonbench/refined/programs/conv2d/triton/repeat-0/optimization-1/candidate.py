import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_C": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_C": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_C": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_C": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_C": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_C": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_C": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_C": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_C": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_C": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_C": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_C": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_C": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_C": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 16, "BLOCK_C": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_C": 32}, num_warps=8, num_stages=2),
    ],
    key=["total_positions"],
)
@triton.jit
def _conv2d_3x3_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    total_positions,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    positions = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    output_channels = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    position_mask = positions < total_positions
    channel_mask = output_channels < 128

    batch = positions // 196
    spatial = positions - batch * 196
    output_row = spatial // 14
    output_col = spatial - output_row * 14

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    channels = tl.arange(0, BLOCK_C)

    for channel_start in tl.static_range(0, 64, BLOCK_C):
        channel_offsets = channel_start + channels
        channel_mask_k = channel_offsets < 64

        for kernel_row in tl.static_range(0, 3):
            for kernel_col in tl.static_range(0, 3):
                input_offsets = (
                    batch[:, None] * 16384
                    + channel_offsets[None, :] * 256
                    + (output_row[:, None] + kernel_row) * 16
                    + output_col[:, None]
                    + kernel_col
                )
                input_mask = position_mask[:, None] & channel_mask_k[None, :]
                input_tile = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)

                weight_offsets = (
                    output_channels[:, None] * 576
                    + channel_offsets[None, :] * 9
                    + kernel_row * 3
                    + kernel_col
                )
                weight_mask = channel_mask[:, None] & channel_mask_k[None, :]
                weight_tile = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0)

                accumulator = tl.dot(
                    input_tile,
                    tl.trans(weight_tile),
                    accumulator,
                    input_precision="ieee",
                )

    if HAS_BIAS:
        bias_tile = tl.load(bias_ptr + output_channels, mask=channel_mask, other=0.0)
        accumulator += bias_tile[None, :]

    output_offsets = (
        batch[:, None] * 25088
        + output_channels[None, :] * 196
        + output_row[:, None] * 14
        + output_col[:, None]
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

        total_positions = batch_size * output_height * output_width
        bias_ptr = bias if bias is not None else input
        grid = lambda META: (
            triton.cdiv(total_positions, META["BLOCK_M"]),
            triton.cdiv(output_channels, META["BLOCK_N"]),
        )
        _conv2d_3x3_kernel[grid](
            input,
            weight,
            bias_ptr,
            output,
            total_positions,
            HAS_BIAS=bias is not None,
        )
        return output

    return wrapper
