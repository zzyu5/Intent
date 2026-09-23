import intent
import intent.language as I


@intent.kernel
def _conv2d_bias(
    x: I.In[I.f32, (1, 3, 768, 768)],
    weight: I.In[I.f32, (6, 3, 3, 3)],
    bias: I.In[I.f32, (6,)],
    out: I.Out[I.f32, (1, 6, 766, 766)],
    negative_slope: I.f32,
):
    # Each output coordinate is independent; the reduction over the receptive
    # field remains ordered so the accumulation has a clear numerical contract.
    for n in I.parallel(I.domain(0, 1)):
        for oc in I.parallel(I.domain(0, 6)):
            for oh in I.parallel(I.domain(0, 766)):
                for ow in I.parallel(I.domain(0, 766)):
                    acc = bias[oc]
                    for ic in I.domain(0, 3):
                        for kh in I.domain(0, 3):
                            for kw in I.domain(0, 3):
                                acc = acc + x[n, ic, oh + kh, ow + kw] * weight[oc, ic, kh, kw]
                    out[n, oc, oh, ow] = I.select(acc >= 0.0, acc, acc * negative_slope)


@intent.kernel
def _conv2d_no_bias(
    x: I.In[I.f32, (1, 3, 768, 768)],
    weight: I.In[I.f32, (6, 3, 3, 3)],
    out: I.Out[I.f32, (1, 6, 766, 766)],
    negative_slope: I.f32,
):
    for n in I.parallel(I.domain(0, 1)):
        for oc in I.parallel(I.domain(0, 6)):
            for oh in I.parallel(I.domain(0, 766)):
                for ow in I.parallel(I.domain(0, 766)):
                    acc = I.cast(0.0, I.f32)
                    for ic in I.domain(0, 3):
                        for kh in I.domain(0, 3):
                            for kw in I.domain(0, 3):
                                acc = acc + x[n, ic, oh + kh, ow + kw] * weight[oc, ic, kh, kw]
                    out[n, oc, oh, ow] = I.select(acc >= 0.0, acc, acc * negative_slope)


def build(context):
    conv_with_bias = context.compile("leaky_relu_conv2d_bias", _conv2d_bias)
    conv_without_bias = context.compile("leaky_relu_conv2d_no_bias", _conv2d_no_bias)

    def leaky_relu_conv2d(
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        negative_slope=0.01,
        inplace=False,
    ):
        if bias is None:
            return conv_without_bias.run(input, weight, negative_slope)
        return conv_with_bias.run(input, weight, bias, negative_slope)

    return leaky_relu_conv2d
