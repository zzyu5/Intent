import intent
import intent.language as I


@intent.kernel
def rsqrt_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    output[elements] = I.fdiv(1.0, input[elements] ** 0.5)


def build(context):
    compiled = context.compile("rsqrt_kernel", rsqrt_kernel)

    def wrapper(input, *, out=None):
        if out is None:
            return compiled.run(input)
        compiled(input, out)
        return out

    return wrapper
