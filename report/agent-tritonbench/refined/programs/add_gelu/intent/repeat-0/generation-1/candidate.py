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
    x = I.cast(input[elements], I.f32)
    y = I.cast(other[elements], I.f32)
    value = x + alpha * y

    cubic = value * value * value
    inner = 0.7978845608028654 * (value + 0.044715 * cubic)
    activated = 0.5 * value * (1.0 + I.tanh(inner))
    output[elements] = I.cast(activated, I.f16)


def build(context):
    compiled = context.compile("add_gelu", add_gelu_kernel)

    def wrapper(input, other, alpha=1, approximate="none", out=None):
        if out is None:
            return compiled.run(input, other, alpha)
        compiled(input, other, alpha, out)
        return out

    return wrapper
