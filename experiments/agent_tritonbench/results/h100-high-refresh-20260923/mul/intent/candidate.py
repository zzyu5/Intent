import torch
import intent
import intent.language as I


@intent.kernel
def _mul_kernel(
    input: I.In[I.f16, ("N",)],
    other: I.In[I.f16, ("N",)],
    out: I.Out[I.f16, ("N",)],
):
    n, = input.shape
    elements = I.domain(0, n)
    for index in I.parallel(elements):
        out[index] = input[index] * other[index]


def build(context):
    mul_artifact = context.compile("mul_elementwise", _mul_kernel)

    def mul(input, other, out=None):
        if out is None:
            out = torch.empty_like(input)
        mul_artifact(input, other, out)
        return out

    return mul
