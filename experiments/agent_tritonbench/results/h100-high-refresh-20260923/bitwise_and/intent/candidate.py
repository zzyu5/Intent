import torch
import intent
import intent.language as I


@intent.kernel
def _bitwise_and_kernel(
    input: I.In[I.i32, ("N",)],
    other: I.In[I.i32, ("N",)],
    out: I.Out[I.i32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    for index in I.parallel(elements):
        out[index] = input[index] & other[index]


def build(context):
    kernel = context.compile("bitwise_and_elementwise", _bitwise_and_kernel)

    def bitwise_and(input, other, out=None):
        if out is None:
            out = torch.empty_like(input)
        kernel(input, other, out)
        return out

    return bitwise_and
