import intent
import intent.language as I


@intent.kernel
def leaky_relu_kernel(
    input: I.In[I.f32, ("N",)],
    negative_slope: I.f32,
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    values = input[elements]
    zero = I.cast(0, I.f32)
    output[elements] = I.maximum(values, zero) + negative_slope * I.minimum(values, zero)


def build(context):
    compiled = context.compile("leaky_relu", leaky_relu_kernel)

    def wrapper(input, negative_slope=0.01, inplace=False):
        if inplace:
            compiled(input, negative_slope, input)
            return input
        return compiled.run(input, negative_slope)

    return wrapper
