import torch
import intent
import intent.language as I


@intent.kernel
def add_kernel(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
    alpha: I.f32,
):
    n = input.shape[0]
    elements = I.domain(0, n)
    out[elements] = input[elements] + alpha * other[elements]


def build(context):
    add_artifact = context.compile("add", add_kernel)

    def add(input, other, alpha=1, out=None):
        result = out if out is not None else torch.empty_like(input)
        add_artifact(input, other, result, float(alpha))
        return result

    return add
