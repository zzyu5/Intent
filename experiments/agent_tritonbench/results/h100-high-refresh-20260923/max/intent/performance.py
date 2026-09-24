import collections

import torch
import intent
import intent.language as I


MaxResult = collections.namedtuple("max", ["values", "indices"])


@intent.kernel
def _max_1d(
    input: I.In[I.f32, ("N",)],
    values: I.Out[I.f32, ()],
    indices: I.Out[I.i64, ()],
):
    value, index = I.arg_reduce.max(input, axis=0)
    values[()] = value
    indices[()] = I.cast(index, I.i64)


def build(context):
    max_1d = context.compile("max_intent_1d", _max_1d, constexprs={})

    def max(input, dim, keepdim=False):
        values = torch.empty((), device=input.device, dtype=input.dtype)
        indices = torch.empty((), device=input.device, dtype=torch.int64)
        max_1d(input, values, indices)
        return MaxResult(values, indices)

    return max
