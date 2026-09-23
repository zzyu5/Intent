import torch
import intent
import intent.language as I


@intent.kernel
def _leaky_relu_out(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
    negative_slope: I.f32,
):
    elements = I.domain(0, input.shape[0])
    for i in I.parallel(elements):
        value = input[i]
        output[i] = I.select(value < 0, value * negative_slope, value)


@intent.kernel
def _leaky_relu_inplace(
    input: I.InOut[I.f32, ("N",)],
    negative_slope: I.f32,
):
    elements = I.domain(0, input.shape[0])
    for i in I.parallel(elements):
        value = input[i]
        input[i] = I.select(value < 0, value * negative_slope, value)


def build(context):
    out_kernel = context.compile("leaky_relu_out", _leaky_relu_out, constexprs={})
    inplace_kernel = context.compile(
        "leaky_relu_inplace", _leaky_relu_inplace, constexprs={}
    )

    def leaky_relu(input, negative_slope=0.01, inplace=False):
        if inplace:
            inplace_kernel(input, negative_slope)
            return input

        output = torch.empty_like(input)
        out_kernel(input, output, negative_slope)
        return output

    return leaky_relu
