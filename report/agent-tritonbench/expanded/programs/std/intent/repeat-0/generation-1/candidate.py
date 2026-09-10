import intent
import intent.language as I


@intent.kernel
def std_kernel(
    x: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ()],
):
    elements = I.domain(0, x.shape[0])
    values = x[elements]
    count = I.cast(x.shape[0], I.f32)

    total = I.reduce.sum(values, axis=0)
    mean = I.fdiv(total, count)
    centered = values - mean
    squared = centered * centered
    sum_squared = I.reduce.sum(squared, axis=0)
    variance = I.fdiv(sum_squared, count - 1.0)

    output[()] = I.sqrt(variance)


def build(context):
    compiled = context.compile("std", std_kernel)

    def wrapper(input, dim=None, *, correction=1, keepdim=False, out=None):
        if out is None:
            return compiled.run(input)
        compiled(input, out)
        return out

    return wrapper
