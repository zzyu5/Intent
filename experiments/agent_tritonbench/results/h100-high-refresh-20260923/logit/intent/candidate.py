import torch
import intent
import intent.language as I


@intent.kernel
def _logit_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n, = input.shape
    elements = I.domain(0, n)
    values = input[elements]
    one = I.cast(1.0, I.f32)
    output[elements] = I.log(I.fdiv(values, one - values))


@intent.kernel
def _logit_clamped_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
    eps: I.f32,
):
    n, = input.shape
    elements = I.domain(0, n)
    values = input[elements]
    one = I.cast(1.0, I.f32)
    upper = one - eps
    values = I.minimum(I.maximum(values, eps), upper)
    output[elements] = I.log(I.fdiv(values, one - values))


def build(context):
    logit = context.compile("logit", _logit_kernel)
    logit_clamped = context.compile("logit_clamped", _logit_clamped_kernel)

    def wrapper(input, eps=None, *, out=None):
        if out is None:
            out = torch.empty_like(input)
        if eps is None:
            logit(input, out)
        else:
            logit_clamped(input, out, eps)
        return out

    return wrapper
