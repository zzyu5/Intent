import intent
import intent.language as I


@intent.kernel
def std_all(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ()],
):
    elements = I.domain(0, input.shape[0])
    values = input[elements]

    count = I.cast(input.shape[0], I.f32)
    total = I.reduce.sum(values, axis=0, acc_dtype=I.f32)
    mean = I.fdiv(total, count)

    centered = values - mean
    squared_deviations = centered * centered
    sum_squared_deviations = I.reduce.sum(
        squared_deviations, axis=0, acc_dtype=I.f32
    )
    variance = I.fdiv(sum_squared_deviations, count - 1.0)
    output[()] = variance ** 0.5


def build(context):
    compiled = context.compile("std_all", std_all)

    def wrapper(input, dim=None, *, correction=1, keepdim=False, out=None):
        if out is None:
            return compiled.run(input)
        compiled(input, out)
        return out

    return wrapper
