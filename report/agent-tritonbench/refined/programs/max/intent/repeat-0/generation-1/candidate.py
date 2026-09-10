from collections import namedtuple

import intent
import intent.language as I


MaxResult = namedtuple("max", ("values", "indices"))


@intent.kernel
def max_reduce(
    input: I.In[I.f32, ("N",)],
    values: I.Out[I.f32, (1,)],
    indices: I.Out[I.i64, (1,)],
):
    reduced = I.arg_reduce.max(input, axis=0)
    output_domain = I.domain(0, 1)
    values[output_domain] = reduced[0]
    indices[output_domain] = I.cast(reduced[1], I.i64)


def build(context):
    compiled = context.compile("max_reduce", max_reduce)

    def wrapper(input, dim, keepdim=False, *, out=None):
        if out is None:
            values, indices = compiled.run(input)
        else:
            values, indices = out
            compiled(input, values.reshape(1), indices.reshape(1))

        if not keepdim:
            values = values.reshape(())
            indices = indices.reshape(())
        return MaxResult(values, indices)

    return wrapper
