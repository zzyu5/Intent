import torch
import intent
import intent.language as I


ELEMENTS = 1048576
GROUPS = 1024
SEGMENT = ELEMENTS // GROUPS


@intent.kernel
def argmax_partials(
    input_tensor: I.In[I.f16, (ELEMENTS,)],
    values: I.Out[I.f16, (GROUPS,)],
    indices: I.Out[I.i64, (GROUPS,)],
):
    elements = I.domain(0, ELEMENTS)
    for group in I.parallel(I.domain(0, GROUPS)):
        begin = group * SEGMENT
        segment = elements[begin:begin + SEGMENT]
        maximum, relative = I.arg_reduce.max(input_tensor[segment], axis=0)
        values[group] = maximum
        indices[group] = I.cast(begin + relative, I.i64)


@intent.kernel
def argmax_finish(
    values: I.In[I.f16, (GROUPS,)],
    indices: I.In[I.i64, (GROUPS,)],
    output: I.Out[I.i64, ()],
):
    _, group = I.arg_reduce.max(values[:], axis=0)
    I.assume_in_bounds(group, indices, axis=0)
    output[()] = indices[group]


def build(context):
    partials = context.compile("argmax_partials", argmax_partials)
    finish = context.compile("argmax_finish", argmax_finish)

    def argmax(input_tensor, dim, keepdim=False):
        values = torch.empty((GROUPS,), dtype=input_tensor.dtype, device=input_tensor.device)
        indices = torch.empty((GROUPS,), dtype=torch.int64, device=input_tensor.device)
        output = torch.empty((), dtype=torch.int64, device=input_tensor.device)
        partials(input_tensor, values, indices)
        finish(values, indices, output)
        return output

    return argmax
