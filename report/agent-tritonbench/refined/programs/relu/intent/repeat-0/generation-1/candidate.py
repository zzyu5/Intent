import intent
import intent.language as I


@intent.kernel
def relu_out(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    output[elements] = I.maximum(input[elements], 0.0)


@intent.kernel
def relu_inplace(input: I.InOut[I.f32, ("N",)]):
    elements = I.domain(0, input.shape[0])
    input[elements] = I.maximum(input[elements], 0.0)


def build(context):
    compiled_out = context.compile("relu_out", relu_out)
    compiled_inplace = context.compile("relu_inplace", relu_inplace)

    def wrapper(input, inplace=False):
        if inplace:
            compiled_inplace.run(input)
            return input
        return compiled_out.run(input)

    return wrapper
