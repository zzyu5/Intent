import intent
import intent.language as I


@intent.kernel
def relu_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    output[elements] = I.maximum(input[elements], 0.0)


def build(context):
    compiled = context.compile("relu", relu_kernel)

    def wrapper(input, inplace=False):
        if inplace:
            compiled(input, input)
            return input
        return compiled.run(input)

    return wrapper
