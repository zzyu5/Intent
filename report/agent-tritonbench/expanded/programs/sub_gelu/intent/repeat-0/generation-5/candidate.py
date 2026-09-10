import intent
import intent.language as I


@intent.kernel
def sub_gelu_exact(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    x = input[elements] - other[elements]

    # Hastings' normal-CDF approximation has sub-1e-7 absolute error over the
    # useful range and maps directly to GELU(x) = x * Phi(x).
    nonnegative = I.cast(x >= 0.0, I.f32)
    sign = nonnegative * 2.0 - 1.0
    abs_x = x * sign
    t = 1.0 / (1.0 + 0.2316419 * abs_x)
    polynomial = (((((1.330274429 * t - 1.821255978) * t + 1.781477937) * t
                    - 0.356563782) * t + 0.319381530) * t)
    normal_pdf = 0.3989422804014327 * I.exp2(-0.5 * abs_x * abs_x * 1.4426950408889634)
    positive_cdf = 1.0 - normal_pdf * polynomial
    cdf = 0.5 + sign * (positive_cdf - 0.5)
    output[elements] = x * cdf


def build(context):
    compiled = context.compile("sub_gelu_exact", sub_gelu_exact)

    def wrapper(input, other, alpha=1, approximate="none", out=None):
        if alpha != 1:
            raise ValueError("the supplied profile requires alpha=1")
        if approximate != "none":
            raise ValueError("the supplied profile requires approximate='none'")
        if out is None:
            return compiled.run(input, other)
        compiled(input, other, out)
        return out

    return wrapper
