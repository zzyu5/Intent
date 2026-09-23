import torch
import intent
import intent.language as I


@intent.kernel
def _sub_kernel(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
    alpha: I.f32,
):
    elements = I.domain(0, input.shape[0])
    for i in I.parallel(elements):
        out[i] = input[i] - alpha * other[i]


def build(context):
    kernel = context.compile("sub_pointwise", _sub_kernel)

    def sub(input: torch.Tensor, other: torch.Tensor, *, alpha: float = 1, out: torch.Tensor = None):
        result = torch.empty_like(input) if out is None else out
        kernel(input, other, result, float(alpha))
        return result

    return sub
