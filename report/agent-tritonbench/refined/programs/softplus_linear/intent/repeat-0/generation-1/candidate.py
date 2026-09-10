import intent
import intent.language as I


@intent.fn
def _stable_softplus(value):
    # log1p(exp(-abs(x))) via the atanh series, with exp2 as the only
    # transcendental required by the DSL.
    magnitude = I.maximum(value, -value)
    z = I.exp2(-magnitude * 1.4426950408889634)
    u = I.fdiv(z, 2.0 + z)
    u2 = u * u
    correction = 1.0 + u2 * (
        0.3333333333333333
        + u2
        * (
            0.2
            + u2
            * (
                0.14285714285714285
                + u2
                * (
                    0.1111111111111111
                    + u2
                    * (
                        0.09090909090909091
                        + u2
                        * (
                            0.07692307692307693
                            + u2 * 0.06666666666666667
                        )
                    )
                )
            )
        )
    )
    log1p_exp = (2.0 * u) * correction
    smooth = I.maximum(value, 0.0) + log1p_exp
    return value if value > 20.0 else smooth


@intent.kernel
def softplus_linear_with_bias(
    input: I.In[I.f32, ("B", "K")],
    weight: I.In[I.f32, ("N", "K")],
    bias: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("B", "N")],
):
    linear = I.matmul(input, weight, acc_dtype=I.f32, transpose_rhs=True)
    linear = linear + bias
    values = _stable_softplus(linear)
    rows = I.domain(0, output.shape[0])
    columns = I.domain(0, output.shape[1])
    output[rows, columns] = values


@intent.kernel
def softplus_linear_without_bias(
    input: I.In[I.f32, ("B", "K")],
    weight: I.In[I.f32, ("N", "K")],
    output: I.Out[I.f32, ("B", "N")],
):
    linear = I.matmul(input, weight, acc_dtype=I.f32, transpose_rhs=True)
    values = _stable_softplus(linear)
    rows = I.domain(0, output.shape[0])
    columns = I.domain(0, output.shape[1])
    output[rows, columns] = values


def build(context):
    with_bias = context.compile("softplus_linear_with_bias", softplus_linear_with_bias)
    without_bias = context.compile(
        "softplus_linear_without_bias", softplus_linear_without_bias
    )

    def wrapper(input, weight, bias=None, beta=1, threshold=20):
        if bias is None:
            return without_bias.run(input, weight)
        return with_bias.run(input, weight, bias)

    return wrapper
