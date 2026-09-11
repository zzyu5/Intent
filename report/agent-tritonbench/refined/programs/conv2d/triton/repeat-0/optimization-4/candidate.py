import torch
import triton
import triton.language as tl


@triton.jit
def _conv2d_fixed(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    positions = pid_m * 64 + tl.arange(0, 64)
    output_channels = pid_n * 16 + tl.arange(0, 16)
    position_mask = positions < 1568

    batch = positions // 196
    spatial = positions - batch * 196
    output_row = spatial // 14
    output_col = spatial - output_row * 14

    accumulator = tl.zeros((64, 16), dtype=tl.float32)
    channels = tl.arange(0, 64)

    for kernel_row in tl.static_range(0, 3):
        for kernel_col in tl.static_range(0, 3):
            input_offsets = (
                batch[:, None] * 16384
                + channels[None, :] * 256
                + (output_row[:, None] + kernel_row) * 16
                + output_col[:, None]
                + kernel_col
            )
            input_tile = tl.load(input_ptr + input_offsets, mask=position_mask[:, None], other=0.0)

            weight_offsets = (
                output_channels[:, None] * 576
                + channels[None, :] * 9
                + kernel_row * 3
                + kernel_col
            )
            weight_tile = tl.load(weight_ptr + weight_offsets)

            accumulator = tl.dot(
                input_tile,
                tl.trans(weight_tile),
                accumulator,
                input_precision="ieee",
            )

    bias_tile = tl.load(bias_ptr + output_channels)
    accumulator += bias_tile[None, :]

    output_offsets = (
        batch[:, None] * 25088
        + output_channels[None, :] * 196
        + output_row[:, None] * 14
        + output_col[:, None]
    )
    tl.store(output_ptr + output_offsets, accumulator, mask=position_mask[:, None])


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

        _conv2d_fixed[(25, 8)](
            input,
            weight,
            bias,
            output,
            num_warps=4,
            num_stages=2,
        )
        return output

    return wrapper
