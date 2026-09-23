import torch
import intent
import intent.language as I


@intent.kernel
def _mul_sub_kernel(
    input: I.In[I.f32, ("N",)],
    other_mul: I.In[I.f32, ("N",)],
    other_sub: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
    alpha: I.f32,
):
    n = input.shape[0]
    elements = I.domain(0, n)
    for i in I.parallel(elements):
        out[i] = input[i] * other_mul[i] - alpha * other_sub[i]


def build(context):
    kernel = context.compile("mul_sub_elementwise", _mul_sub_kernel)

    def mul_sub(input, other_mul, other_sub, alpha=1, out=None):
        result = out if out is not None else torch.empty_like(input)
        kernel(input, other_mul, other_sub, result, alpha)
        return result

    return mul_sub
