import intent
import intent.language as I


@intent.kernel
def add_mean_kernel(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ()],
):
    elements = I.domain(0, input.shape[0])
    total = I.reduce.sum(input[elements] + other[elements], axis=0)
    output[()] = total / I.cast(input.shape[0], I.f32)


def build(context):
    compiled = context.compile("add_mean", add_mean_kernel)

    def wrapper(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
        return compiled.run(input, other)

    return wrapper
