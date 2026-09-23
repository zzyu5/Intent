import torch
import intent
import intent.language as I


@intent.kernel
def _logsumexp_max(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ()],
):
    n = input.shape[0]
    elements = I.domain(0, n)
    values = input[elements]
    maximum = I.reduce.max(values, axis=0, acc_dtype=I.f32)
    output[()] = maximum


@intent.kernel
def _logsumexp_sum(
    input: I.In[I.f32, ("N",)],
    maximum_input: I.In[I.f32, ()],
    output: I.Out[I.f32, ()],
):
    maximum = maximum_input[()]
    positive_infinity = I.cast(I.inf, I.f32)

    if maximum == positive_infinity:
        output[()] = positive_infinity
    elif maximum == -positive_infinity:
        output[()] = -positive_infinity
    else:
        n = input.shape[0]
        elements = I.domain(0, n)
        values = input[elements]
        shifted = values - maximum
        terms = I.exp(shifted)
        total = I.reduce.sum(terms, axis=0, acc_dtype=I.f32)
        output[()] = I.log(total) + maximum


def build(context):
    max_kernel = context.compile("logsumexp_max", _logsumexp_max)
    sum_kernel = context.compile("logsumexp_sum", _logsumexp_sum)

    def logsumexp(input, dim, keepdim=False, *, out=None):
        del dim, keepdim
        maximum = torch.empty((), device=input.device, dtype=input.dtype)
        result = out if out is not None else torch.empty((), device=input.device, dtype=input.dtype)
        max_kernel(input, maximum)
        sum_kernel(input, maximum, result)
        return result

    return logsumexp
