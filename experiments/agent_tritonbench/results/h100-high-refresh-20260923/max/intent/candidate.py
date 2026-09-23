import collections

import torch
import intent
import intent.language as I


@intent.kernel
def _argmax_1d(
    x: I.In[I.f32, ("N",)],
    values: I.Out[I.f32, ()],
    indices: I.Out[I.i64, ()],
):
    domain = I.domain(0, x.shape[0])
    value = x[domain]
    maximum, position = I.arg_reduce.max(value, axis=0)
    values[()] = maximum
    indices[()] = I.cast(position, I.i64)


def build(context):
    argmax = context.compile("argmax_1d", _argmax_1d)
    result_type = collections.namedtuple("max", ("values", "indices"))

    def max(input, dim, keepdim=False):
        values = torch.empty((), dtype=input.dtype, device=input.device)
        indices = torch.empty((), dtype=torch.int64, device=input.device)
        argmax(input, values, indices)
        return result_type(values, indices)

    return max
