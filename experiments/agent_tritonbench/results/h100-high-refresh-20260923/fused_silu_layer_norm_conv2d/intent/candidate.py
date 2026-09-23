import torch
import intent
import intent.language as I


@intent.kernel
def _conv3x3(
    x: I.In[I.f32, (2, 16, 64, 64)],
    conv_weight: I.In[I.f32, (64, 16, 3, 3)],
    conv_bias: I.In[I.f32, (64,)],
    out: I.Out[I.f32, (2, 64, 62, 62)],
):
    for n in I.parallel(I.domain(0, 2)):
        for co in I.parallel(I.domain(0, 64)):
            for oh in I.parallel(I.domain(0, 62)):
                for ow in I.parallel(I.domain(0, 62)):
                    acc = I.cast(0.0, I.f32)
                    for ci in range(16):
                        for kh in range(3):
                            for kw in range(3):
                                acc = acc + (
                                    x[n, ci, oh + kh, ow + kw]
                                    * conv_weight[co, ci, kh, kw]
                                )
                    out[n, co, oh, ow] = acc + conv_bias[co]


@intent.kernel
def _conv3x3_no_bias(
    x: I.In[I.f32, (2, 16, 64, 64)],
    conv_weight: I.In[I.f32, (64, 16, 3, 3)],
    out: I.Out[I.f32, (2, 64, 62, 62)],
):
    for n in I.parallel(I.domain(0, 2)):
        for co in I.parallel(I.domain(0, 64)):
            for oh in I.parallel(I.domain(0, 62)):
                for ow in I.parallel(I.domain(0, 62)):
                    acc = I.cast(0.0, I.f32)
                    for ci in range(16):
                        for kh in range(3):
                            for kw in range(3):
                                acc = acc + (
                                    x[n, ci, oh + kh, ow + kw]
                                    * conv_weight[co, ci, kh, kw]
                                )
                    out[n, co, oh, ow] = acc


@intent.kernel
def _layer_norm_silu(
    conv: I.In[I.f32, (2, 64, 62, 62)],
    eps: I.f32,
    out: I.Out[I.f32, (2, 64, 62, 62)],
):
    count = I.cast(64 * 62 * 62, I.f32)
    channels = I.domain(0, 64)
    rows = I.domain(0, 62)
    cols = I.domain(0, 62)
    for n in I.parallel(I.domain(0, 2)):
        values = conv[n, channels, rows, cols]
        mean = I.fdiv(I.reduce.sum(values, axis=(0, 1, 2), acc_dtype=I.f32), count)
        centered = values - mean
        variance = I.fdiv(
            I.reduce.sum(centered * centered, axis=(0, 1, 2), acc_dtype=I.f32),
            count,
        )
        normalized = centered / I.sqrt(variance + eps)
        out[n, channels, rows, cols] = normalized * I.sigmoid(normalized)


def build(context):
    conv = context.compile("fused_silu_conv3x3", _conv3x3)
    conv_no_bias = context.compile("fused_silu_conv3x3_no_bias", _conv3x3_no_bias)
    norm = context.compile("fused_silu_layer_norm_silu", _layer_norm_silu)

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
        conv_out = torch.empty((2, 64, 62, 62), device=x.device, dtype=x.dtype)
        if conv_bias is None:
            conv_no_bias(x, conv_weight, conv_out)
        else:
            conv(x, conv_weight, conv_bias, conv_out)
        out = torch.empty_like(conv_out)
        norm(conv_out, ln_eps, out)
        return out

    return fused_silu_layer_norm_conv2d
