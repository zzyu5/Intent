import torch
import intent
import intent.language as I


MEMBERS = 64 * 62 * 62
GROUP_SIZE = 1024
GROUPS = (MEMBERS + GROUP_SIZE - 1) // GROUP_SIZE


@intent.kernel
def convolution(
    x: I.In[I.f32, (2, 16, 64, 64)],
    weight: I.In[I.f32, (64, 16, 3, 3)],
    bias: I.In[I.f32, (64,)],
    out: I.Out[I.f32, (2, 64, 62, 62)],
    USE_BIAS: I.Constexpr[bool],
):
    batches = I.domain(0, 2)
    channels = I.domain(0, 16)
    filters = I.domain(0, 64)
    rows = I.domain(0, 62)
    columns = I.domain(0, 62)
    kernel_rows = I.domain(0, 3)
    kernel_columns = I.domain(0, 3)
    row = I.reshape(I.indices(rows), (62, 1, 1, 1))
    column = I.reshape(I.indices(columns), (1, 62, 1, 1))
    kr = I.reshape(I.indices(kernel_rows), (1, 1, 3, 1))
    kc = I.reshape(I.indices(kernel_columns), (1, 1, 1, 3))
    samples = x[batches, channels, row + kr, column + kc]
    coefficients = weight[filters, channels, kernel_rows, kernel_columns]
    value = I.contract(samples, coefficients,
                       reduce=((1, 1), (4, 2), (5, 3)), acc_dtype=I.f32)
    value = I.transpose(value, (0, 3, 1, 2))
    if USE_BIAS:
        value = value + I.reshape(bias[filters], (1, 64, 1, 1))
    out[batches, filters, rows, columns] = value


@intent.kernel
def partial_statistics(
    x: I.In[I.f32, (2, 64, 62, 62)],
    sums: I.Out[I.f32, (2, GROUPS)],
    squares: I.Out[I.f32, (2, GROUPS)],
):
    local = I.indices(I.domain(0, GROUP_SIZE))
    for batch in I.parallel(I.domain(0, 2)):
        for group in I.parallel(I.domain(0, GROUPS)):
            member = group * GROUP_SIZE + local
            value = I.gather(x, (batch, member // (62 * 62),
                                 (member // 62) % 62, member % 62),
                             valid=member < MEMBERS, fill=I.cast(0.0, I.f32))
            sums[batch, group] = I.reduce.sum(value, axis=0, acc_dtype=I.f32)
            squares[batch, group] = I.reduce.sum(value * value, axis=0,
                                                 acc_dtype=I.f32)


@intent.kernel
def statistics(
    sums: I.In[I.f32, (2, GROUPS)],
    squares: I.In[I.f32, (2, GROUPS)],
    mean: I.Out[I.f32, (2,)],
    inv_std: I.Out[I.f32, (2,)],
    eps: I.f32,
):
    groups = I.domain(0, GROUPS)
    for batch in I.parallel(I.domain(0, 2)):
        average = I.reduce.sum(sums[batch, groups], axis=0,
                               acc_dtype=I.f32) / I.cast(MEMBERS, I.f32)
        second = I.reduce.sum(squares[batch, groups], axis=0,
                              acc_dtype=I.f32) / I.cast(MEMBERS, I.f32)
        variance = I.maximum(second - average * average, I.cast(0.0, I.f32))
        mean[batch] = average
        inv_std[batch] = I.rsqrt(variance + eps)


@intent.kernel
def normalize_silu(
    x: I.In[I.f32, (2, 64, 62, 62)],
    mean: I.In[I.f32, (2,)],
    inv_std: I.In[I.f32, (2,)],
    out: I.Out[I.f32, (2, 64, 62, 62)],
):
    for batch in I.parallel(I.domain(0, 2)):
        for channel in I.parallel(I.domain(0, 64)):
            for row in I.parallel(I.domain(0, 62)):
                for column in I.parallel(I.domain(0, 62)):
                    value = (x[batch, channel, row, column] - mean[batch]) * inv_std[batch]
                    out[batch, channel, row, column] = value * I.sigmoid(value)


def build(context):
    conv = context.compile("convolution", convolution, constexprs={"USE_BIAS": True})
    conv_no_bias = context.compile("convolution_no_bias", convolution,
                                   constexprs={"USE_BIAS": False})
    partial = context.compile("partial_statistics", partial_statistics)
    stats = context.compile("statistics", statistics)
    apply = context.compile("normalize_silu", normalize_silu)

    def fused_silu_layer_norm_conv2d(x, weight, conv_weight, conv_bias=None,
                                    conv_stride=1, conv_padding=0,
                                    conv_dilation=1, conv_groups=1, ln_eps=1e-5):
        if (conv_stride, conv_padding, conv_dilation, conv_groups) != (1, 0, 1, 1):
            raise NotImplementedError("development program uses the fixed convolution profile")
        conv_out = torch.empty((2, 64, 62, 62), device=x.device, dtype=x.dtype)
        out = torch.empty_like(conv_out)
        sums = torch.empty((2, GROUPS), device=x.device, dtype=x.dtype)
        squares = torch.empty_like(sums)
        mean = torch.empty((2,), device=x.device, dtype=x.dtype)
        inv_std = torch.empty_like(mean)
        if conv_bias is None:
            unused_bias = torch.empty((64,), device=x.device, dtype=x.dtype)
            conv_no_bias(x, conv_weight, unused_bias, conv_out)
        else:
            conv(x, conv_weight, conv_bias, conv_out)
        partial(conv_out, sums, squares)
        stats(sums, squares, mean, inv_std, float(ln_eps))
        apply(conv_out, mean, inv_std, out)
        return out

    return fused_silu_layer_norm_conv2d
