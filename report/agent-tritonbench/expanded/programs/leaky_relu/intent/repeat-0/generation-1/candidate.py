import intent
import intent.language as I


@intent.kernel
def leaky_relu_out(
    x: I.In[I.f32, ("N",)],
    negative_slope: I.f32,
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, x.shape[0])
    values = x[elements]
    output[elements] = I.maximum(values, 0.0) + negative_slope * I.minimum(values, 0.0)


@intent.kernel
def leaky_relu_inplace(
    x: I.InOut[I.f32, ("N",)],
    negative_slope: I.f32,
):
    elements = I.domain(0, x.shape[0])
    values = x[elements]
    x[elements] = I.maximum(values, 0.0) + negative_slope * I.minimum(values, 0.0)


def build(context):
    compiled_out = context.compile("leaky_relu_out", leaky_relu_out)
    compiled_inplace = context.compile("leaky_relu_inplace", leaky_relu_inplace)

    def wrapper(input, negative_slope=0.01, inplace=False):
        if inplace:
            compiled_inplace.run(input, negative_slope)
            return input
        return compiled_out.run(input, negative_slope)

    return wrapper
