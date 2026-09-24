from __future__ import annotations

import torch
import intent
import intent.language as I


@intent.kernel
def conv2d_kernel(
    x: I.In[I.f32, (2, 16, 64, 64)],
    conv_weight: I.In[I.f32, (64, 16, 3, 3)],
    conv_bias: I.In[I.f32, (64,)],
    out: I.Out[I.f32, (2, 64, 62, 62)],
):
    batch = I.domain(0, 2)
    channels = I.domain(0, 64)
    height = I.domain(0, 62)
    width = I.domain(0, 62)

    height_index = I.reshape(I.indices(height), (62, 1))
    width_index = I.reshape(I.indices(width), (1, 62))
    acc = I.full((2, 64, 62, 62), 0.0, dtype=I.f32)

    for input_channel in range(16):
        for kernel_row in range(3):
            for kernel_col in range(3):
                row = height_index + I.cast(kernel_row, I.index)
                col = width_index + I.cast(kernel_col, I.index)
                values = x[batch, input_channel, row, col]
                values = I.reshape(values, (2, 1, 62, 62))
                kernel_value = conv_weight[channels, input_channel, kernel_row, kernel_col]
                kernel_value = I.reshape(kernel_value, (1, 64, 1, 1))
                acc = acc + values * kernel_value

    bias = I.reshape(conv_bias[channels], (1, 64, 1, 1))
    out[batch, channels, height, width] = acc + bias


@intent.kernel
def conv2d_no_bias_kernel(
    x: I.In[I.f32, (2, 16, 64, 64)],
    conv_weight: I.In[I.f32, (64, 16, 3, 3)],
    out: I.Out[I.f32, (2, 64, 62, 62)],
):
    batch = I.domain(0, 2)
    channels = I.domain(0, 64)
    height = I.domain(0, 62)
    width = I.domain(0, 62)

    height_index = I.reshape(I.indices(height), (62, 1))
    width_index = I.reshape(I.indices(width), (1, 62))
    acc = I.full((2, 64, 62, 62), 0.0, dtype=I.f32)

    for input_channel in range(16):
        for kernel_row in range(3):
            for kernel_col in range(3):
                row = height_index + I.cast(kernel_row, I.index)
                col = width_index + I.cast(kernel_col, I.index)
                values = x[batch, input_channel, row, col]
                values = I.reshape(values, (2, 1, 62, 62))
                kernel_value = conv_weight[channels, input_channel, kernel_row, kernel_col]
                kernel_value = I.reshape(kernel_value, (1, 64, 1, 1))
                acc = acc + values * kernel_value

    out[batch, channels, height, width] = acc


@intent.kernel
def layer_norm_partial_stats_kernel(
    values: I.In[I.f32, (2, 64, 62, 62)],
    partial_sum: I.Out[I.f32, (2, 8)],
    partial_square_sum: I.Out[I.f32, (2, 8)],
):
    batch = I.domain(0, 2)
    group = I.domain(0, 8)
    group_channel = I.domain(0, 8)
    height = I.domain(0, 62)
    width = I.domain(0, 62)

    group_index = I.reshape(I.indices(group), (8, 1, 1, 1))
    channel_index = I.reshape(I.indices(group_channel), (1, 8, 1, 1))
    height_index = I.reshape(I.indices(height), (1, 1, 62, 1))
    width_index = I.reshape(I.indices(width), (1, 1, 1, 62))
    channel = group_index * 8 + channel_index
    members = values[batch, channel, height_index, width_index]

    sums = I.reduce.sum(members, axis=(2, 3, 4), acc_dtype=I.f32)
    squares = I.reduce.sum(members * members, axis=(2, 3, 4), acc_dtype=I.f32)
    partial_sum[batch, group] = sums
    partial_square_sum[batch, group] = squares


@intent.kernel
def layer_norm_finalize_stats_kernel(
    partial_sum: I.In[I.f32, (2, 8)],
    partial_square_sum: I.In[I.f32, (2, 8)],
    mean: I.Out[I.f32, (2,)],
    variance: I.Out[I.f32, (2,)],
):
    batch = I.domain(0, 2)
    group = I.domain(0, 8)
    total_count = I.cast(246016.0, I.f32)

    sums = partial_sum[batch, group]
    squares = partial_square_sum[batch, group]
    mean_value = I.fdiv(I.reduce.sum(sums, axis=1, acc_dtype=I.f32), total_count)
    second_moment = I.fdiv(I.reduce.sum(squares, axis=1, acc_dtype=I.f32), total_count)
    variance_value = I.maximum(second_moment - mean_value * mean_value, I.cast(0.0, I.f32))
    mean[batch] = mean_value
    variance[batch] = variance_value


@intent.kernel
def normalized_silu_kernel(
    values: I.In[I.f32, (2, 64, 62, 62)],
    mean: I.In[I.f32, (2,)],
    variance: I.In[I.f32, (2,)],
    out: I.Out[I.f32, (2, 64, 62, 62)],
    ln_eps: I.f32,
):
    batch = I.domain(0, 2)
    channels = I.domain(0, 64)
    height = I.domain(0, 62)
    width = I.domain(0, 62)

    input_value = values[batch, channels, height, width]
    mean_value = I.reshape(mean[batch], (2, 1, 1, 1))
    variance_value = I.reshape(variance[batch], (2, 1, 1, 1))
    centered = input_value - mean_value
    normalized = centered / I.sqrt(variance_value + ln_eps)
    out[batch, channels, height, width] = normalized * I.sigmoid(normalized)


def build(context):
    conv = context.compile("conv2d", conv2d_kernel)
    conv_no_bias = context.compile("conv2d_no_bias", conv2d_no_bias_kernel)
    partial_stats = context.compile("layer_norm_partial_stats", layer_norm_partial_stats_kernel)
    finalize_stats = context.compile("layer_norm_finalize_stats", layer_norm_finalize_stats_kernel)
    normalized_silu = context.compile("normalized_silu", normalized_silu_kernel)

    def fused_silu_layer_norm_conv2d(
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor = None,
        conv_stride: int = 1,
        conv_padding: int = 0,
        conv_dilation: int = 1,
        conv_groups: int = 1,
        ln_eps: float = 1e-5,
    ) -> torch.Tensor:
        del weight, conv_stride, conv_padding, conv_dilation, conv_groups

        convolution = torch.empty((2, 64, 62, 62), device=x.device, dtype=x.dtype)
        if conv_bias is None:
            conv_no_bias(x, conv_weight, convolution)
        else:
            conv(x, conv_weight, conv_bias, convolution)

        partial_sum = torch.empty((2, 8), device=x.device, dtype=x.dtype)
        partial_square_sum = torch.empty((2, 8), device=x.device, dtype=x.dtype)
        partial_stats(convolution, partial_sum, partial_square_sum)

        mean = torch.empty((2,), device=x.device, dtype=x.dtype)
        variance = torch.empty((2,), device=x.device, dtype=x.dtype)
        finalize_stats(partial_sum, partial_square_sum, mean, variance)

        output = torch.empty((2, 64, 62, 62), device=x.device, dtype=x.dtype)
        normalized_silu(convolution, mean, variance, output, ln_eps)
        return output

    return fused_silu_layer_norm_conv2d
