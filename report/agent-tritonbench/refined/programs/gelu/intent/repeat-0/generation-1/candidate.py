import intent
import intent.language as I


@intent.kernel
def gelu_none(
    x: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, x.shape[0])
    value = x[elements]

    # Approximate erf(x / sqrt(2)) with the Abramowitz-Stegun polynomial.
    scaled = value * 0.7071067811865475
    magnitude = I.maximum(scaled, -scaled)
    t = I.fdiv(1.0, 1.0 + 0.3275911 * magnitude)
    polynomial = (((((1.061405429 * t - 1.453152027) * t
                     + 1.421413741) * t - 0.284496736) * t
                   + 0.254829592) * t)
    decay = I.exp2(-magnitude * magnitude * 1.4426950408889634)
    erf_magnitude = 1.0 - polynomial * decay
    erf_value = (erf_magnitude if scaled >= 0.0 else -erf_magnitude)
    output[elements] = 0.5 * value * (1.0 + erf_value)


@intent.kernel
def gelu_tanh(
    x: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, x.shape[0])
    value = x[elements]
    cube = value * value * value
    argument = 0.7978845608028654 * (value + 0.044715 * cube)
    output[elements] = 0.5 * value * (1.0 + I.tanh(argument))


def build(context):
    exact = context.compile("gelu_none", gelu_none)
    tanh = context.compile("gelu_tanh", gelu_tanh)

    def wrapper(input, approximate="none"):
        if approximate == "none":
            return exact.run(input)
        if approximate == "tanh":
            return tanh.run(input)
        raise ValueError("approximate must be 'none' or 'tanh'")

    return wrapper
