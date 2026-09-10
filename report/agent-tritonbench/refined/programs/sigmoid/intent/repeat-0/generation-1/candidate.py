import intent
import intent.language as I


@intent.kernel
def sigmoid_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    output[elements] = 0.5 * (I.tanh(0.5 * input[elements]) + 1.0)


def build(context):
    compiled = context.compile("sigmoid", sigmoid_kernel)

    def wrapper(input, *, out=None):
        if out is None:
            return compiled.run(input)
        compiled(input, out)
        return out

    return wrapper
