import torch
import intent
import intent.language as I


@intent.kernel
def abs_kernel(
    input_tensor: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n = input_tensor.shape[0]
    elements = I.domain(0, n)
    for index in I.parallel(elements):
        output[index] = I.abs(input_tensor[index])


def build(context):
    artifact = context.compile("abs_1d", abs_kernel)

    def abs(input_tensor, out=None):
        if out is None:
            out = torch.empty_like(input_tensor)
        artifact(input_tensor, out)
        return out

    return abs
