import math

import torch
import intent
import intent.language as I


@intent.kernel
def _conv2d_add_kernel(
    input: I.In[I.f32, ("N", "CI", "IH", "IW")],
    weight: I.In[I.f32, ("CO", "CG", "KH", "KW")],
    other: I.In[I.f32, ("N", "CO", "OH", "OW")],
    out: I.Out[I.f32, ("N", "CO", "OH", "OW")],
    stride_h: I.i64,
    stride_w: I.i64,
    pad_h: I.i64,
    pad_w: I.i64,
    dilation_h: I.i64,
    dilation_w: I.i64,
    groups: I.i64,
    alpha: I.f32,
):
    n_size, _, in_h, in_w = input.shape
    out_channels, channels_per_group, kernel_h, kernel_w = weight.shape
    out_h, out_w = out.shape[2], out.shape[3]

    sh = I.cast(stride_h, I.index)
    sw = I.cast(stride_w, I.index)
    ph = I.cast(pad_h, I.index)
    pw = I.cast(pad_w, I.index)
    dh = I.cast(dilation_h, I.index)
    dw = I.cast(dilation_w, I.index)
    group_count = I.cast(groups, I.index)
    output_channels_per_group = out_channels // group_count

    for n in I.parallel(I.domain(0, n_size)):
        for co in I.parallel(I.domain(0, out_channels)):
            group = co // output_channels_per_group
            input_channel_start = group * channels_per_group
            for oh in I.parallel(I.domain(0, out_h)):
                for ow in I.parallel(I.domain(0, out_w)):
                    acc = I.cast(0.0, I.f32)
                    for ci in I.domain(0, channels_per_group):
                        input_channel = input_channel_start + ci
                        for kh in I.domain(0, kernel_h):
                            input_h = oh * sh - ph + kh * dh
                            for kw in I.domain(0, kernel_w):
                                input_w = ow * sw - pw + kw * dw
                                sample = I.cast(0.0, I.f32)
                                if input_h >= 0:
                                    if input_h < in_h:
                                        if input_w >= 0:
                                            if input_w < in_w:
                                                sample = input[n, input_channel, input_h, input_w]
                                acc = acc + sample * weight[co, ci, kh, kw]
                    out[n, co, oh, ow] = acc + alpha * other[n, co, oh, ow]


def _pair(value):
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    value = int(value)
    return value, value


def _padding(value, input_h, input_w, kernel_h, kernel_w, stride_h, stride_w):
    if isinstance(value, str):
        if value == "valid":
            return 0, 0
        if value == "same":
            return (kernel_h - 1) // 2, (kernel_w - 1) // 2
        raise ValueError("unsupported padding mode")
    return _pair(value)


def build(context):
    compiled = context.compile("conv2d_add", _conv2d_add_kernel)

    def conv2d_add(
        input,
        weight,
        bias=None,
        other=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        alpha=1,
        out=None,
    ):
        if bias is not None:
            raise NotImplementedError("the supplied profile has no bias")
        if other is None:
            raise NotImplementedError("the supplied profile provides other")
        if not isinstance(other, torch.Tensor):
            raise TypeError("the supplied profile provides tensor other")

        stride_h, stride_w = _pair(stride)
        dilation_h, dilation_w = _pair(dilation)
        kernel_h, kernel_w = int(weight.shape[2]), int(weight.shape[3])
        pad_h, pad_w = _padding(
            padding,
            int(input.shape[2]),
            int(input.shape[3]),
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        )
        out_h = (int(input.shape[2]) + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        out_w = (int(input.shape[3]) + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

        if out is None:
            out = torch.empty(
                (int(input.shape[0]), int(weight.shape[0]), out_h, out_w),
                device=input.device,
                dtype=input.dtype,
            )

        compiled(
            input,
            weight,
            other,
            out,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            int(groups),
            float(alpha),
        )
        return out

    return conv2d_add
