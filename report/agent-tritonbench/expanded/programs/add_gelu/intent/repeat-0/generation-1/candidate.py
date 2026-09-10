import intent
import intent.language as I


@intent.kernel
def add_gelu_kernel(
    input: I.In[I.f16, ("N",)],
    other: I.In[I.f16, ("N",)],
    alpha: I.f32,
    output: I.Out[I.f16, ("N",)],
):
    elements = I.domain(0, input.shape[0])

    x = I.cast(input[elements], I.f32) + alpha * I.cast(other[elements], I.f32)
    x_squared = x * x
    x_cubed = x_squared * x
    tanh_argument = 0.7978845608028654 * (x + 0.044715 * x_cubed)
    result = 0.5 * x * (1.0 + I.tanh(tanh_argument))
    output[elements] = I.cast(result, I.f16)


def build(context):
    compiled = context.compile("add_gelu", add_gelu_kernel)

    def wrapper(input, other, alpha=1, approximate="none", out=None):
        if out is None:
            return compiled.run(input, other, alpha)
        compiled(input, other, alpha, out)
        return out

    return wrapper
