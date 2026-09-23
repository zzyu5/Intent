import torch
import intent
import intent.language as I


@intent.kernel
def _gelu_exact(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n = input.shape[0]
    elements = I.domain(0, n)
    value = input[elements]

    half = I.cast(0.5, I.f32)
    one = I.cast(1.0, I.f32)
    inv_sqrt_two = I.cast(0.7071067811865476, I.f32)
    output[elements] = half * value * (one + I.erf(value * inv_sqrt_two))


@intent.kernel
def _gelu_tanh(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n = input.shape[0]
    elements = I.domain(0, n)
    value = input[elements]

    half = I.cast(0.5, I.f32)
    one = I.cast(1.0, I.f32)
    coefficient = I.cast(0.044715, I.f32)
    sqrt_two_over_pi = I.cast(0.7978845608028654, I.f32)
    cubic = value * value * value
    argument = sqrt_two_over_pi * (value + coefficient * cubic)
    output[elements] = half * value * (
        one + I.tanh(argument, approximate=True)
    )


def build(context):
    exact = context.compile("gelu_exact", _gelu_exact)
    tanh = context.compile("gelu_tanh", _gelu_tanh)

    def gelu(input: torch.Tensor, approximate: str = "none") -> torch.Tensor:
        output = torch.empty_like(input)
        if approximate == "none":
            exact(input, output)
        elif approximate == "tanh":
            tanh(input, output)
        else:
            raise ValueError("approximate must be 'none' or 'tanh'")
        return output

    return gelu
