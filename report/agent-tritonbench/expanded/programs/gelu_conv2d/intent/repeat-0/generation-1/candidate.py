import intent
import intent.language as I


@intent.fn
def _gelu_none(x: I.f32):
    # Abramowitz-Stegun 7.1.26 gives a sufficiently accurate erf approximation
    # using only the pointwise operations available in the Intent surface.
    if x < 0.0:
        ax = -x
    else:
        ax = x

    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
              - 0.284496736) * t + 0.254829592) * t)
    erf_abs = 1.0 - poly * I.exp2(-ax * ax * 1.4426950408889634)

    if x < 0.0:
        erf_x = -erf_abs
    else:
        erf_x = erf_abs
    return 0.5 * x * (1.0 + erf_x)


@intent.kernel
def _gelu_conv2d(
    input: I.In[I.f32, ("N", "IC", "H", "W")],
    weight: I.In[I.f32, ("OC", "IC", "KH", "KW")],
    output: I.Out[I.f32, ("N", "OC", "H", "W")],
):
    batches = I.domain(0, input.shape[0])
    output_channels = I.domain(0, weight.shape[0])
    output_rows = I.domain(0, output.shape[2])
    output_columns = I.domain(0, output.shape[3])

    for n in I.parallel(batches):
        for oc in I.parallel(output_channels):
            for oh in I.parallel(output_rows):
                for ow in I.parallel(output_columns):
                    acc = I.cast(0.0, I.f32)
                    for ic in I.domain(0, input.shape[1]):
                        for kh in I.domain(0, weight.shape[2]):
                            ih = oh + kh - 1
                            if ih >= 0:
                                if ih < input.shape[2]:
                                    for kw in I.domain(0, weight.shape[3]):
                                        iw = ow + kw - 1
                                        if iw >= 0:
                                            if iw < input.shape[3]:
                                                acc = acc + input[n, ic, ih, iw] * weight[oc, ic, kh, kw]
                    output[n, oc, oh, ow] = _gelu_none(acc)


def build(context):
    compiled = context.compile("gelu_conv2d_nobias", _gelu_conv2d, constexprs={})

    def wrapper(
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        approximate="none",
        out=None,
    ):
        if bias is not None:
            raise NotImplementedError("the fixed invocation profile has no bias")
        if stride != 1 or padding != 1 or dilation != 1 or groups != 1 or approximate != "none":
            raise NotImplementedError("unsupported parameters for the fixed invocation profile")
        if out is None:
            return compiled.run(input, weight)
        compiled(input, weight, out)
        return out

    return wrapper
