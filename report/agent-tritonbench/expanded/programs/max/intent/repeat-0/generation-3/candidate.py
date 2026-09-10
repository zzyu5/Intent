import math
from collections import namedtuple

import intent
import intent.language as I


@intent.kernel
def max_1d(
    input: I.In[I.f32, ("N",)],
    values: I.Out[I.f32, (1,)],
    indices: I.Out[I.i64, (1,)],
):
    result = I.arg_reduce.max(
        input,
        axis=0,
        identity=(I.cast(-math.inf, I.f32), I.cast(9223372036854775807, I.i64)),
    )
    values[0] = result[0]
    indices[0] = result[1]


_MaxResult = namedtuple("max", ("values", "indices"))


def build(context):
    compiled = context.compile("max_1d", max_1d)

    def wrapper(input, dim, keepdim=False, *, out=None):
        if dim not in (0, -1):
            raise ValueError("dim must be 0 for a one-dimensional input")

        if out is None:
            values, indices = compiled.run(input)
            values = values.reshape((1,)) if keepdim else values.reshape(())
            indices = indices.reshape((1,)) if keepdim else indices.reshape(())
            return _MaxResult(values, indices)

        values_out, indices_out = out
        values_arg = values_out.reshape((1,))
        indices_arg = indices_out.reshape((1,))
        compiled(input, values_arg, indices_arg)
        return _MaxResult(values_out, indices_out)

    return wrapper
