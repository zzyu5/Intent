import intent
import intent.language as I


@intent.kernel
def exp_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    output[elements] = I.exp2(input[elements] * 1.4426950408889634)


def build(context):
    compiled = context.compile("exp_kernel", exp_kernel)

    def wrapper(input, *, out=None):
        if out is None:
            return compiled.run(input)
        compiled(input, out)
        return out

    return wrapper
