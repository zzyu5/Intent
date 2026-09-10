import intent
import intent.language as I


@intent.kernel
def log1p_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    x = input[elements]
    one_plus_x = x + 1.0

    # Reduce the positive argument to [1, 2), where the atanh series
    # converges quickly.  The correction preserves x when 1 + x rounds.
    bits = I.bitcast(one_plus_x, I.u32)
    exponent = (bits >> 23) & 255
    mantissa_bits = (bits & 0x7FFFFF) | 0x3F800000
    mantissa = I.bitcast(mantissa_bits, I.f32)
    z = I.fdiv(mantissa - 1.0, mantissa + 1.0)
    z2 = z * z
    log_mantissa = 2.0 * z * (
        1.0
        + z2
        * (
            1.0 / 3.0
            + z2
            * (
                1.0 / 5.0
                + z2 * (1.0 / 7.0 + z2 * (1.0 / 9.0 + z2 / 11.0))
            )
        )
    )
    log_value = (I.cast(exponent, I.f32) - 127.0) * 0.6931471805599453 + log_mantissa
    rounding_correction = I.fdiv(x - (one_plus_x - 1.0), one_plus_x)
    finite_result = log_value + rounding_correction

    # Preserve log1p's domain and infinity behavior without a dedicated
    # logarithm intrinsic.
    nan_value = I.fdiv(0.0, x - x)
    negative_infinity = I.fdiv(-1.0, one_plus_x)
    positive_infinity = one_plus_x
    is_special = exponent == 255
    special_result = positive_infinity if one_plus_x > 0.0 else nan_value
    result = special_result if is_special else finite_result
    result = negative_infinity if x == -1.0 else result
    result = nan_value if x < -1.0 else result
    output[elements] = result


def build(context):
    compiled = context.compile("log1p", log1p_kernel)

    def wrapper(input, *, out=None):
        if out is None:
            return compiled.run(input)
        compiled(input, out)
        return out

    return wrapper
