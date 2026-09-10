import intent
import intent.language as I


@intent.kernel
def mul_kernel(
    input: I.In[I.f16, ("N",)],
    other: I.In[I.f16, ("N",)],
    output: I.Out[I.f16, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    output[elements] = input[elements] * other[elements]


def build(context):
    compiled = context.compile("mul", mul_kernel)

    def wrapper(input, other, *, out=None):
        if out is None:
            return compiled.run(input, other)
        compiled(input, other, out)
        return out

    return wrapper
