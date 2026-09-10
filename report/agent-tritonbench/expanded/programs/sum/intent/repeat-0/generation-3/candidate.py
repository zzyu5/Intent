import intent
import intent.language as I


@intent.kernel
def sum_partial(
    input: I.In[I.f32, (1048576,)],
    output: I.Out[I.f32, (256,)],
):
    source = I.domain(0, 1048576)
    parts = I.domain(0, 256)
    for part in I.parallel(parts):
        begin = part * 4096
        end = (part + 1) * 4096
        region = source[begin:end]
        output[part] = I.reduce.sum(input[region], axis=0)


@intent.kernel
def sum_finalize(
    input: I.In[I.f32, (256,)],
    output: I.Out[I.f32, (1,)],
):
    output[0] = I.reduce.sum(input, axis=0)


def build(context):
    partial = context.compile("sum_partial_3", sum_partial)
    finalize = context.compile("sum_finalize_3", sum_finalize)

    def wrapper(input, dim, keepdim=False, *, dtype=None):
        partials = partial.run(input)
        return finalize.run(partials).view(())

    return wrapper
