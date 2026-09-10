import intent
import intent.language as I


@intent.kernel
def sum_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ()],
):
    output[()] = I.reduce.sum(input, axis=0)


def build(context):
    compiled = context.compile("sum_kernel", sum_kernel)

    def wrapper(input, dim, keepdim=False, *, dtype=None):
        return compiled.run(input)

    return wrapper
