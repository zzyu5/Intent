import intent
import intent.language as I


@intent.kernel
def asin_kernel(
    input: I.In[I.f16, ("N",)],
    output: I.Out[I.f16, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    x = I.cast(input[elements], I.f32)
    absolute_x = I.maximum(x, -x)

    # The square root in the standard fast asin approximation is evaluated
    # with a fixed Newton iteration in f32.  The initial value is positive
    # for every in-range input, including the endpoint at one.
    remainder = 1.0 - absolute_x
    root = remainder + 1.0
    for _ in range(10):
        root = 0.5 * (root + I.fdiv(remainder, root, approximate=True))
    root = I.select(remainder == 0.0, 0.0, root)

    polynomial = ((-0.0187293 * absolute_x + 0.0742610) * absolute_x - 0.2121144)
    polynomial = polynomial * absolute_x + 1.5707288
    magnitude = 1.5707963267948966 - root * polynomial
    signed = I.select(x < 0.0, -magnitude, magnitude)

    zero = x - x
    nan = I.fdiv(zero, zero)
    signed = I.select(x < -1.0, nan, signed)
    signed = I.select(x > 1.0, nan, signed)
    output[elements] = I.cast(signed, I.f16)


def build(context):
    compiled = context.compile("asin", asin_kernel)

    def wrapper(input, *, out=None):
        if out is None:
            return compiled.run(input)
        compiled(input, out)
        return out

    return wrapper
