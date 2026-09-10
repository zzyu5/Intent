import intent
import intent.language as I


@intent.kernel
def abs_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    values = input[elements]
    output[elements] = I.maximum(values, -values)


def build(context):
    compiled = context.compile("abs", abs_kernel)

    def wrapper(input, *, out=None):
        if out is None:
            return compiled.run(input)
        compiled(input, out)
        return out

    return wrapper
