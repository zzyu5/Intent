import torch
import intent
import intent.language as I


@intent.kernel
def mul_relu_kernel(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    output[elements] = I.maximum(input[elements] * other[elements], 0.0)


def build(context):
    compiled = context.compile("mul_relu", mul_relu_kernel)

    def wrapper(input, other, inplace=False, out=None):
        if out is not None:
            output = out
        elif inplace:
            output = input
        else:
            output = torch.empty_like(input)

        compiled(input, other, output)
        return output

    return wrapper
