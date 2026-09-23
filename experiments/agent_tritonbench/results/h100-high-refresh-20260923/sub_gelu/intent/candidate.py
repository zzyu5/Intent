import torch
import intent
import intent.language as I


@intent.kernel
def gelu_kernel(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
    alpha: I.f32,
    APPROXIMATE: I.Constexpr[bool],
):
    n, = input.shape
    elements = I.domain(0, n)
    half = I.cast(0.5, I.f32)
    sqrt_two = I.cast(1.4142135623730951, I.f32)
    tanh_scale = I.cast(0.7978845608028654, I.f32)
    cubic_coeff = I.cast(0.044715, I.f32)

    for i in I.parallel(elements):
        z = input[i] - alpha * other[i]
        if APPROXIMATE:
            z3 = z * z * z
            result = half * z * (I.cast(1.0, I.f32) + I.tanh(tanh_scale * (z + cubic_coeff * z3)))
        else:
            result = half * z * I.erf(z / sqrt_two)
        output[i] = result


def build(context):
    exact = context.compile(
        "sub_gelu_none",
        gelu_kernel,
        constexprs={"APPROXIMATE": False},
    )
    tanh = context.compile(
        "sub_gelu_tanh",
        gelu_kernel,
        constexprs={"APPROXIMATE": True},
    )

    def sub_gelu(input, other, alpha=1, approximate="none", out=None):
        if out is None:
            out = torch.empty_like(input)
        artifact = tanh if approximate == "tanh" else exact
        artifact(input, other, out, alpha)
        return out

    return sub_gelu
