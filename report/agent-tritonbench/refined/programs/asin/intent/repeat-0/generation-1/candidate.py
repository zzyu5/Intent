import intent
import intent.language as I


@intent.kernel
def asin_kernel(
    input: I.In[I.f16, ("N",)],
    output: I.Out[I.f16, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    output[elements] = I.asin(input[elements])


def build(context):
    compiled = context.compile("asin_f16", asin_kernel)

    def wrapper(input, *, out=None):
        if out is None:
            return compiled.run(input)
        compiled(input, out)
        return out

    return wrapper
