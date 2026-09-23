import torch
import intent
import intent.language as I


@intent.kernel
def divide_kernel(
    input: I.In[I.f32, ("M",)],
    other: I.In[I.f32, ("M",)],
    output: I.Out[I.f32, ("M",)],
):
    M, = input.shape
    for index in I.parallel(I.domain(0, M)):
        output[index] = I.fdiv(input[index], other[index])


def build(context):
    divide = context.compile("divide_elementwise", divide_kernel)

    def div(input, other, *, rounding_mode=None, out=None):
        if rounding_mode is not None:
            raise NotImplementedError("only the default true-division mode is profiled")
        result = out if out is not None else torch.empty_like(input)
        divide(input, other, result)
        return result

    return div
