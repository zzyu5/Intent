import intent
import intent.language as I


@intent.kernel
def argmax_kernel(
    input: I.In[I.f16, ("N",)],
    output: I.Out[I.i64, (1,)],
):
    output[0] = I.arg_reduce.max(input, axis=0, identity=-1)


def build(context):
    compiled = context.compile("argmax_kernel", argmax_kernel)

    def wrapper(input, dim, keepdim=False):
        return compiled.run(input).view(())

    return wrapper
