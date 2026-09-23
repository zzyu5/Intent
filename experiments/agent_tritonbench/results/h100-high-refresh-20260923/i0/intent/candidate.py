import torch
import intent
import intent.language as I


@intent.kernel
def i0_kernel(
    input_tensor: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n = input_tensor.shape[0]
    elements = I.domain(0, n)
    for index in I.parallel(elements):
        output[index] = I.i0(input_tensor[index])


def build(context):
    compiled = context.compile("i0_pointwise", i0_kernel)

    def i0(input_tensor, out=None):
        result = out if out is not None else torch.empty_like(input_tensor)
        compiled(input_tensor, result)
        return result

    return i0
