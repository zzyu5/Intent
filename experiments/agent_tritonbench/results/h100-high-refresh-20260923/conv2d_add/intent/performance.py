import torch
import intent
import intent.language as I


@intent.kernel
def _conv_stage(
    x: I.In[I.f32, (16, 32, 32, 32)],
    weight: I.In[I.f32, (32, 32, 3, 3)],
    result: I.Out[I.f32, (16, 32, 32, 32)],
):
    """Compute the convolution for each independent output element."""
    for n in I.parallel(I.domain(0, 16)):
        for oc in I.parallel(I.domain(0, 32)):
            for oh in I.parallel(I.domain(0, 32)):
                for ow in I.parallel(I.domain(0, 32)):
                    acc = I.cast(0.0, I.f32)
                    for ic in I.domain(0, 32):
                        for kh in I.domain(0, 3):
                            for kw in I.domain(0, 3):
                                ih = oh + kh - 1
                                iw = ow + kw - 1
                                if ih >= 0:
                                    if ih < 32:
                                        if iw >= 0:
                                            if iw < 32:
                                                acc = acc + x[n, ic, ih, iw] * weight[oc, ic, kh, kw]
                    result[n, oc, oh, ow] = acc


@intent.kernel
def _epilogue_tensor(
    convolution: I.In[I.f32, (16, 32, 32, 32)],
    other: I.In[I.f32, (16, 32, 32, 32)],
    result: I.Out[I.f32, (16, 32, 32, 32)],
    alpha: I.f32,
):
    for n in I.parallel(I.domain(0, 16)):
        for oc in I.parallel(I.domain(0, 32)):
            for oh in I.parallel(I.domain(0, 32)):
                for ow in I.parallel(I.domain(0, 32)):
                    value = convolution[n, oc, oh, ow]
                    result[n, oc, oh, ow] = value + alpha * other[n, oc, oh, ow]


@intent.kernel
def _epilogue_tensor_bias(
    convolution: I.In[I.f32, (16, 32, 32, 32)],
    bias: I.In[I.f32, (32,)],
    other: I.In[I.f32, (16, 32, 32, 32)],
    result: I.Out[I.f32, (16, 32, 32, 32)],
    alpha: I.f32,
):
    for n in I.parallel(I.domain(0, 16)):
        for oc in I.parallel(I.domain(0, 32)):
            for oh in I.parallel(I.domain(0, 32)):
                for ow in I.parallel(I.domain(0, 32)):
                    value = convolution[n, oc, oh, ow] + bias[oc]
                    result[n, oc, oh, ow] = value + alpha * other[n, oc, oh, ow]


@intent.kernel
def _epilogue_scalar(
    convolution: I.In[I.f32, (16, 32, 32, 32)],
    result: I.Out[I.f32, (16, 32, 32, 32)],
    other: I.f32,
    alpha: I.f32,
):
    for n in I.parallel(I.domain(0, 16)):
        for oc in I.parallel(I.domain(0, 32)):
            for oh in I.parallel(I.domain(0, 32)):
                for ow in I.parallel(I.domain(0, 32)):
                    result[n, oc, oh, ow] = convolution[n, oc, oh, ow] + alpha * other


@intent.kernel
def _epilogue_scalar_bias(
    convolution: I.In[I.f32, (16, 32, 32, 32)],
    bias: I.In[I.f32, (32,)],
    result: I.Out[I.f32, (16, 32, 32, 32)],
    other: I.f32,
    alpha: I.f32,
):
    for n in I.parallel(I.domain(0, 16)):
        for oc in I.parallel(I.domain(0, 32)):
            for oh in I.parallel(I.domain(0, 32)):
                for ow in I.parallel(I.domain(0, 32)):
                    value = convolution[n, oc, oh, ow] + bias[oc]
                    result[n, oc, oh, ow] = value + alpha * other


@intent.kernel
def _copy_stage(
    source: I.In[I.f32, (16, 32, 32, 32)],
    result: I.Out[I.f32, (16, 32, 32, 32)],
):
    for n in I.parallel(I.domain(0, 16)):
        for oc in I.parallel(I.domain(0, 32)):
            for oh in I.parallel(I.domain(0, 32)):
                for ow in I.parallel(I.domain(0, 32)):
                    result[n, oc, oh, ow] = source[n, oc, oh, ow]


def build(context):
    conv = context.compile("conv2d_add_conv", _conv_stage)
    add_tensor = context.compile("conv2d_add_tensor", _epilogue_tensor)
    add_tensor_bias = context.compile("conv2d_add_tensor_bias", _epilogue_tensor_bias)
    add_scalar = context.compile("conv2d_add_scalar", _epilogue_scalar)
    add_scalar_bias = context.compile("conv2d_add_scalar_bias", _epilogue_scalar_bias)
    copy = context.compile("conv2d_add_copy", _copy_stage)

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
        del stride, padding, dilation, groups
        result = out if out is not None else torch.empty_like(input)
        convolution = torch.empty_like(result)
        conv(input, weight, convolution)

        if other is None:
            if bias is None:
                copy(convolution, result)
            else:
                add_scalar_bias(convolution, bias, result, 0.0, 0.0)
        elif isinstance(other, torch.Tensor):
            if bias is None:
                add_tensor(convolution, other, result, float(alpha))
            else:
                add_tensor_bias(convolution, bias, other, result, float(alpha))
        else:
            scalar = float(other)
            if bias is None:
                add_scalar(convolution, result, scalar, float(alpha))
            else:
                add_scalar_bias(convolution, bias, result, scalar, float(alpha))
        return result

    return conv2d_add
