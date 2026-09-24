import torch
import intent
import intent.language as I


@intent.kernel
def _conv_with_bias(
    x: I.In[I.f32, (2, 16, 64, 64)],
    conv_weight: I.In[I.f32, (64, 16, 3, 3)],
    conv_bias: I.In[I.f32, (64,)],
    conv_out: I.Out[I.f32, (2, 64, 62, 62)],
    conv_stride: I.i64,
    conv_padding: I.i64,
    conv_dilation: I.i64,
    conv_groups: I.i64,
):
    n = I.domain(0, 2)
    co = I.domain(0, 64)
    oh = I.domain(0, 62)
    ow = I.domain(0, 62)
    ci = I.domain(0, 16)
    kh = I.domain(0, 3)
    kw = I.domain(0, 3)

    # The seven-axis value keeps output coordinates free and the convolution
    # window axes explicit until the final reduction.
    ni = I.reshape(I.indices(n), (2, 1, 1, 1, 1, 1, 1))
    coi = I.reshape(I.indices(co), (1, 64, 1, 1, 1, 1, 1))
    ohi = I.reshape(I.indices(oh), (1, 1, 62, 1, 1, 1, 1))
    owi = I.reshape(I.indices(ow), (1, 1, 1, 62, 1, 1, 1))
    cii = I.reshape(I.indices(ci), (1, 1, 1, 1, 16, 1, 1))
    khi = I.reshape(I.indices(kh), (1, 1, 1, 1, 1, 3, 1))
    kwi = I.reshape(I.indices(kw), (1, 1, 1, 1, 1, 1, 3))

    stride = I.cast(conv_stride, I.index)
    padding = I.cast(conv_padding, I.index)
    dilation = I.cast(conv_dilation, I.index)
    groups = I.cast(conv_groups, I.index)
    out_channels_per_group = I.cast(64, I.index) // groups
    input_channel = (coi // out_channels_per_group) * I.cast(16, I.index) + cii
    input_y = ohi * stride - padding + khi * dilation
    input_x = owi * stride - padding + kwi * dilation
    valid = (
        (input_channel >= 0)
        & (input_channel < 16)
        & (input_y >= 0)
        & (input_y < 64)
        & (input_x >= 0)
        & (input_x < 64)
    )

    samples = I.gather(
        x,
        (ni, input_channel, input_y, input_x),
        valid=valid,
        fill=I.cast(0.0, I.f32),
    )
    kernel = conv_weight[coi, cii, khi, kwi]
    products = samples * kernel
    conv = I.reduce.sum(products, axis=(4, 5, 6), acc_dtype=I.f32)
    bias = I.reshape(conv_bias, (1, 64, 1, 1))
    conv_out[n, co, oh, ow] = conv + bias


@intent.kernel
def _conv_without_bias(
    x: I.In[I.f32, (2, 16, 64, 64)],
    conv_weight: I.In[I.f32, (64, 16, 3, 3)],
    conv_out: I.Out[I.f32, (2, 64, 62, 62)],
    conv_stride: I.i64,
    conv_padding: I.i64,
    conv_dilation: I.i64,
    conv_groups: I.i64,
):
    n = I.domain(0, 2)
    co = I.domain(0, 64)
    oh = I.domain(0, 62)
    ow = I.domain(0, 62)
    ci = I.domain(0, 16)
    kh = I.domain(0, 3)
    kw = I.domain(0, 3)

    ni = I.reshape(I.indices(n), (2, 1, 1, 1, 1, 1, 1))
    coi = I.reshape(I.indices(co), (1, 64, 1, 1, 1, 1, 1))
    ohi = I.reshape(I.indices(oh), (1, 1, 62, 1, 1, 1, 1))
    owi = I.reshape(I.indices(ow), (1, 1, 1, 62, 1, 1, 1))
    cii = I.reshape(I.indices(ci), (1, 1, 1, 1, 16, 1, 1))
    khi = I.reshape(I.indices(kh), (1, 1, 1, 1, 1, 3, 1))
    kwi = I.reshape(I.indices(kw), (1, 1, 1, 1, 1, 1, 3))

    stride = I.cast(conv_stride, I.index)
    padding = I.cast(conv_padding, I.index)
    dilation = I.cast(conv_dilation, I.index)
    groups = I.cast(conv_groups, I.index)
    out_channels_per_group = I.cast(64, I.index) // groups
    input_channel = (coi // out_channels_per_group) * I.cast(16, I.index) + cii
    input_y = ohi * stride - padding + khi * dilation
    input_x = owi * stride - padding + kwi * dilation
    valid = (
        (input_channel >= 0)
        & (input_channel < 16)
        & (input_y >= 0)
        & (input_y < 64)
        & (input_x >= 0)
        & (input_x < 64)
    )
    samples = I.gather(
        x,
        (ni, input_channel, input_y, input_x),
        valid=valid,
        fill=I.cast(0.0, I.f32),
    )
    kernel = conv_weight[coi, cii, khi, kwi]
    conv_out[n, co, oh, ow] = I.reduce.sum(
        samples * kernel, axis=(4, 5, 6), acc_dtype=I.f32
    )


@intent.kernel
def _batch_stats(
    conv: I.In[I.f32, (2, 64, 62, 62)],
    mean: I.Out[I.f32, (2,)],
    variance: I.Out[I.f32, (2,)],
):
    n = I.domain(0, 2)
    c = I.domain(0, 64)
    h = I.domain(0, 62)
    w = I.domain(0, 62)
    values = conv[n, c, h, w]
    count = I.cast(64 * 62 * 62, I.f32)
    means = I.fdiv(I.reduce.sum(values, axis=(1, 2, 3), acc_dtype=I.f32), count)
    centered = values - I.reshape(means, (2, 1, 1, 1))
    variances = I.fdiv(
        I.reduce.sum(centered * centered, axis=(1, 2, 3), acc_dtype=I.f32),
        count,
    )
    mean[n] = means
    variance[n] = variances


@intent.kernel
def _normalize_silu(
    conv: I.In[I.f32, (2, 64, 62, 62)],
    mean: I.In[I.f32, (2,)],
    variance: I.In[I.f32, (2,)],
    out: I.Out[I.f32, (2, 64, 62, 62)],
    ln_eps: I.f32,
):
    n = I.domain(0, 2)
    c = I.domain(0, 64)
    h = I.domain(0, 62)
    w = I.domain(0, 62)
    values = conv[n, c, h, w]
    means = I.reshape(mean[n], (2, 1, 1, 1))
    variances = I.reshape(variance[n], (2, 1, 1, 1))
    normalized = I.fdiv(values - means, I.sqrt(variances + ln_eps))
    out[n, c, h, w] = normalized * I.sigmoid(normalized)


def build(context):
    conv_with_bias = context.compile("fused_conv_with_bias", _conv_with_bias)
    conv_without_bias = context.compile("fused_conv_without_bias", _conv_without_bias)
    stats = context.compile("fused_batch_stats", _batch_stats)
    normalize_silu = context.compile("fused_normalize_silu", _normalize_silu)

    def fused_silu_layer_norm_conv2d(
        x: "torch.Tensor",
        weight: "torch.Tensor",
        conv_weight: "torch.Tensor",
        conv_bias: "torch.Tensor" = None,
        conv_stride: "int" = 1,
        conv_padding: "int" = 0,
        conv_dilation: "int" = 1,
        conv_groups: "int" = 1,
        ln_eps: "float" = 1e-05,
    ) -> "torch.Tensor":
        conv = torch.empty((2, 64, 62, 62), dtype=x.dtype, device=x.device)
        if conv_bias is None:
            conv_without_bias(
                x,
                conv_weight,
                conv,
                conv_stride,
                conv_padding,
                conv_dilation,
                conv_groups,
            )
        else:
            conv_with_bias(
                x,
                conv_weight,
                conv_bias,
                conv,
                conv_stride,
                conv_padding,
                conv_dilation,
                conv_groups,
            )
        mean = torch.empty((2,), dtype=x.dtype, device=x.device)
        variance = torch.empty((2,), dtype=x.dtype, device=x.device)
        stats(conv, mean, variance)
        out = torch.empty((2, 64, 62, 62), dtype=x.dtype, device=x.device)
        normalize_silu(conv, mean, variance, out, ln_eps)
        return out

    return fused_silu_layer_norm_conv2d
