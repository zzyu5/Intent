import torch
import intent
import intent.language as I


ELEMENTS = 1048576
GROUPS = 1024
SEGMENT = ELEMENTS // GROUPS


@intent.kernel
def logsumexp_partials(
    input: I.In[I.f32, (ELEMENTS,)],
    maxima: I.Out[I.f32, (GROUPS,)],
    sums: I.Out[I.f32, (GROUPS,)],
):
    elements = I.domain(0, ELEMENTS)
    for group in I.parallel(I.domain(0, GROUPS)):
        begin = group * SEGMENT
        values = input[elements[begin:begin + SEGMENT]]
        maximum = I.reduce.max(values, axis=0)
        infinity = I.cast(I.inf, I.f32)
        finite_shift = I.select((maximum == infinity) | (maximum == -infinity), 0.0, maximum)
        total = I.reduce.sum(I.exp(values - finite_shift), axis=0)
        maxima[group] = maximum
        sums[group] = I.select(maximum == infinity, 1.0, total)


@intent.kernel
def logsumexp_finish(
    maxima: I.In[I.f32, (GROUPS,)],
    sums: I.In[I.f32, (GROUPS,)],
    output: I.Out[I.f32, ()],
):
    maximum = I.reduce.max(maxima[:], axis=0)
    infinity = I.cast(I.inf, I.f32)
    if maximum == infinity:
        output[()] = infinity
    elif maximum == -infinity:
        output[()] = -infinity
    else:
        total = I.reduce.sum(sums[:] * I.exp(maxima[:] - maximum), axis=0)
        output[()] = maximum + I.log(total)


def build(context):
    partials = context.compile("logsumexp_partials", logsumexp_partials)
    finish = context.compile("logsumexp_finish", logsumexp_finish)

    def logsumexp(input, dim, keepdim=False, *, out=None):
        maxima = torch.empty((GROUPS,), dtype=input.dtype, device=input.device)
        sums = torch.empty_like(maxima)
        output = out if out is not None else torch.empty((), dtype=input.dtype, device=input.device)
        partials(input, maxima, sums)
        finish(maxima, sums, output)
        return output

    return logsumexp
