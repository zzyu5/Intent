import torch
import intent
import intent.language as I


@intent.kernel
def sqrt_exp_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n = input.shape[0]
    elements = I.domain(0, n)
    for index in I.parallel(elements):
        output[index] = I.exp(I.sqrt(input[index]))


def build(context):
    kernel = context.compile("sqrt_exp_kernel", sqrt_exp_kernel)

    def sqrt_exp(input, out=None):
        if out is None:
            out = torch.empty_like(input)
        kernel(input, out)
        return out

    return sqrt_exp
