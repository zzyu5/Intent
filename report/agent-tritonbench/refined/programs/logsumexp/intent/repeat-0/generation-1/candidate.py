import math

import intent
import intent.language as I


@intent.kernel
def logsumexp_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ()],
):
    elements = I.domain(0, input.shape[0])
    maximum = I.reduce.max(input[elements], axis=0)
    shifted = input[elements] - maximum
    total = I.reduce.sum(
        I.exp2(shifted * 1.4426950408889634),
        axis=0,
        acc_dtype=I.f32,
    )
    output[()] = maximum + math.log(total)


def build(context):
    compiled = context.compile("logsumexp", logsumexp_kernel)

    def wrapper(input, dim, keepdim=False, *, out=None):
        if out is None:
            return compiled.run(input)
        compiled(input, out)
        return out

    return wrapper
