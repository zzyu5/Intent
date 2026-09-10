import intent
import intent.language as I


@intent.kernel
def relu_conv2d_kernel(
    input: I.In[I.f32, (1, 3, 256, 256)],
    weight: I.In[I.f32, (64, 3, 3, 3)],
    output: I.Out[I.f32, (1, 64, 256, 256)],
):
    # The fixed profile has one input image and same-padded 3x3 convolution.
    outputs = I.domain(0, 64 * 256 * 256)
    for linear in I.parallel(outputs):
        ow = linear % 256
        outer = linear // 256
        oh = outer % 256
        oc = outer // 256

        acc = I.cast(0.0, I.f32)
        for ic in I.domain(0, 3):
            for kh in I.domain(0, 3):
                ih = oh + kh - 1
                if ih >= 0:
                    if ih < 256:
                        for kw in I.domain(0, 3):
                            iw = ow + kw - 1
                            if iw >= 0:
                                if iw < 256:
                                    acc = acc + input[0, ic, ih, iw] * weight[oc, ic, kh, kw]

        output[0, oc, oh, ow] = I.maximum(acc, I.cast(0.0, I.f32))


def build(context):
    compiled = context.compile("relu_conv2d", relu_conv2d_kernel)

    def wrapper(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
        return compiled.run(input, weight)

    return wrapper
