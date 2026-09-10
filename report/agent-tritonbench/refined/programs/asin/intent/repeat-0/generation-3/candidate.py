import intent
import intent.language as I


@intent.fn
def asin_scalar(value: I.f32) -> I.f32:
    pi_half = 1.5707963267948966
    zero = I.cast(0, I.f32)
    nan = I.fdiv(zero, zero)
    x = value
    negative = x < 0.0
    ax = -x if negative else x

    result = nan
    if ax <= 1.0:
        result = pi_half
        if ax < 1.0:
            # This polynomial is accurate on [0, 1] once multiplied by
            # sqrt(1 - x).  Newton iterations provide that square root.
            radicand = 1.0 - ax
            root = 1.0
            root = 0.5 * (root + I.fdiv(radicand, root))
            root = 0.5 * (root + I.fdiv(radicand, root))
            root = 0.5 * (root + I.fdiv(radicand, root))
            root = 0.5 * (root + I.fdiv(radicand, root))
            root = 0.5 * (root + I.fdiv(radicand, root))
            root = 0.5 * (root + I.fdiv(radicand, root))
            root = 0.5 * (root + I.fdiv(radicand, root))
            root = 0.5 * (root + I.fdiv(radicand, root))
            poly = ((-0.0187293 * ax + 0.0742610) * ax - 0.2121144) * ax + 1.5707288
            result = pi_half - root * poly

    return -result if negative else result


@intent.kernel
def asin_kernel(
    input: I.In[I.f16, ("N",)],
    output: I.Out[I.f16, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    for element in I.parallel(elements):
        value = I.cast(input[element], I.f32)
        output[element] = I.cast(asin_scalar(value), I.f16)


def build(context):
    compiled = context.compile("asin_f16", asin_kernel)

    def wrapper(input, *, out=None):
        if out is None:
            return compiled.run(input)
        compiled(input, out)
        return out

    return wrapper
