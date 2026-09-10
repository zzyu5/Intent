import intent
import intent.language as I


@intent.kernel
def gelu_exact(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    x = input[elements]

    # Abramowitz-Stegun 7.1.26 approximates erf to about 1.5e-7.
    abs_x = I.maximum(x, -x)
    t = I.fdiv(1.0, 1.0 + 0.3275911 * abs_x)
    polynomial = (((((1.061405429 * t - 1.453152027) * t
                     + 1.421413741) * t - 0.284496736) * t
                   + 0.254829592) * t)
    exp_term = I.exp2(-1.4426950408889634 * x * x)
    erf_abs = 1.0 - polynomial * exp_term
    erf_value = I.select(x < 0.0, -erf_abs, erf_abs)
    erf_value = I.select(x == 0.0, x, erf_value)
    output[elements] = 0.5 * x * (1.0 + erf_value)


@intent.kernel
def gelu_tanh(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    x = input[elements]
    x3 = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x3)
    output[elements] = 0.5 * x * (1.0 + I.tanh(inner))


def build(context):
    exact = context.compile("gelu_exact", gelu_exact)
    tanh = context.compile("gelu_tanh", gelu_tanh)

    def wrapper(input, approximate="none"):
        if approximate == "none":
            return exact.run(input)
        if approximate == "tanh":
            return tanh.run(input)
        raise ValueError("approximate must be 'none' or 'tanh'")

    return wrapper
