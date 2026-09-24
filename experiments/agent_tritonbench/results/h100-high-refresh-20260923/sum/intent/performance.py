import torch
import intent
import intent.language as I


ELEMENTS = 1048576
GROUPS = 256
GROUP_SIZE = ELEMENTS // GROUPS


@intent.kernel
def reduce_partials(
    input: I.In[I.f32, (1048576,)],
    partials: I.Out[I.f32, (256,)],
):
    source = I.domain(0, 1048576)
    groups = I.domain(0, 256)

    for group in I.parallel(groups):
        begin = group * 4096
        members = source[begin : begin + 4096]
        values = input[members]
        partials[group] = I.reduce.sum(values, axis=0, acc_dtype=I.f32)


@intent.kernel
def finish_sum(
    partials: I.In[I.f32, (256,)],
    output: I.Out[I.f32, ()],
):
    groups = I.domain(0, 256)
    values = partials[groups]
    output[()] = I.reduce.sum(values, axis=0, acc_dtype=I.f32)


def build(context):
    partial_kernel = context.compile("sum_reduce_partials", reduce_partials)
    finish_kernel = context.compile("sum_finish", finish_sum)

    def sum(input, dim, keepdim=False, dtype=None):
        partials = torch.empty((GROUPS,), device=input.device, dtype=torch.float32)
        output = torch.empty((), device=input.device, dtype=torch.float32)
        partial_kernel(input, partials)
        finish_kernel(partials, output)
        return output

    return sum
