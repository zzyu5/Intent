import torch
import intent
import intent.language as I


@intent.kernel
def _sqrt_tanh_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n = input.shape[0]
    for i in I.parallel(I.domain(0, n)):
        output[i] = I.tanh(I.sqrt(input[i]))


def build(context):
    kernel = context.compile("sqrt_tanh", _sqrt_tanh_kernel)

    def sqrt_tanh(input, out=None):
        if out is None:
            out = torch.empty_like(input)
        kernel(input, out)
        return out

    return sqrt_tanh
