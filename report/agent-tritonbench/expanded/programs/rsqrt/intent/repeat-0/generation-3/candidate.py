import torch
import intent
import intent.language as I


@intent.kernel
def rsqrt_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    for element in elements:
        output[element] = I.rsqrt(input[element])


def build(context):
    compiled = context.compile("rsqrt_generation_3", rsqrt_kernel)

    def wrapper(input, *, out=None):
        if out is None:
            out = torch.empty(input.shape, device=input.device, dtype=input.dtype)
        compiled(input, out)
        return out

    return wrapper
