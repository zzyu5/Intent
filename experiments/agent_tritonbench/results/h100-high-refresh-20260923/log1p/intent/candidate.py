import torch
import intent
import intent.language as I


@intent.kernel
def log1p_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    for index in I.parallel(elements):
        output[index] = I.log1p(input[index])


def build(context):
    compiled = context.compile("log1p", log1p_kernel)

    def log1p(input, *, out=None):
        if out is None:
            out = torch.empty_like(input)
        compiled(input, out)
        return out

    return log1p
