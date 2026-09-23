import torch
import intent
import intent.language as I


@intent.kernel
def _exp_sqrt_kernel(
    x: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
):
    n = x.shape[0]
    points = I.domain(0, n)
    exponent = I.exp(x[points])
    result = I.sqrt(exponent)
    out[points] = result


def build(context):
    kernel = context.compile("exp_sqrt", _exp_sqrt_kernel)

    def exp_sqrt(input, out=None):
        if out is None:
            out = torch.empty_like(input)
        kernel(input, out)
        return out

    return exp_sqrt
