import intent
import intent.language as I


@intent.fn
def _log_from_atanh(z):
    z2 = z * z
    series = 1.0 / 19.0
    series = 1.0 / 17.0 + z2 * series
    series = 1.0 / 15.0 + z2 * series
    series = 1.0 / 13.0 + z2 * series
    series = 1.0 / 11.0 + z2 * series
    series = 1.0 / 9.0 + z2 * series
    series = 1.0 / 7.0 + z2 * series
    series = 1.0 / 5.0 + z2 * series
    series = 1.0 / 3.0 + z2 * series
    return 2.0 * z * (1.0 + z2 * series)


@intent.kernel
def log1p_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    values = input[elements]

    # Use x/(x+2) directly near zero, where forming 1+x would lose bits.
    small_z = I.fdiv(values, values + 2.0, approximate=True)
    small_result = _log_from_atanh(small_z)

    one_plus = values + 1.0
    bits = I.bitcast(one_plus, I.u32)
    exponent_bits = (bits >> 23) & 255
    mantissa_bits = (bits & 0x007FFFFF) | 0x3F800000
    mantissa = I.bitcast(mantissa_bits, I.f32)
    reduced_z = I.fdiv(mantissa - 1.0, mantissa + 1.0, approximate=True)
    reduced_result = (
        I.cast(exponent_bits, I.f32) - 127.0
    ) * 0.6931471805599453 + _log_from_atanh(reduced_z)

    use_small = (values > -0.5) & (values < 0.5)
    output[elements] = I.select(use_small, small_result, reduced_result)


def build(context):
    compiled = context.compile("log1p", log1p_kernel)

    def wrapper(input, *, out=None):
        if out is None:
            return compiled.run(input)
        compiled(input, out)
        return out

    return wrapper
