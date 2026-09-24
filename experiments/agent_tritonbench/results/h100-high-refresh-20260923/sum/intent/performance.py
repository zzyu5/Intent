import intent
import intent.language as I


@intent.kernel
def _sum_all(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ()],
):
    n = input.shape[0]
    values = input[I.domain(0, n)]
    total = I.reduce.sum(values, axis=0, acc_dtype=I.f32)
    output[()] = total


def build(context):
    sum_all = context.compile("sum_all", _sum_all)

    def sum(input, dim, keepdim=False, *, dtype=None):
        return sum_all.run(input)

    return sum
