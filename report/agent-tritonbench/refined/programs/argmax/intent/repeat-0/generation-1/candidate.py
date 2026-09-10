import intent
import intent.language as I


@intent.kernel
def argmax_kernel(
    input: I.In[I.f16, ("N",)],
    output: I.Out[I.i64, ()],
):
    elements = I.domain(0, input.shape[0])
    output[()] = I.arg_reduce.max(input[elements], axis=0)


def build(context):
    compiled = context.compile("argmax", argmax_kernel)

    def wrapper(input, dim, keepdim=False):
        return compiled.run(input)

    return wrapper
