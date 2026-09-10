import intent
import intent.language as I


@intent.kernel
def sum_rows(
    input: I.In[I.f32, (256, 4096)],
    output: I.Out[I.f32, (256,)],
):
    output[:] = I.reduce.sum(input, axis=1)


@intent.kernel
def sum_partials(
    input: I.In[I.f32, (256,)],
    output: I.Out[I.f32, (1,)],
):
    output[0] = I.reduce.sum(input, axis=0)


def build(context):
    rows = context.compile("sum_rows_4", sum_rows)
    partials = context.compile("sum_partials_4", sum_partials)

    def wrapper(input, dim, keepdim=False, *, dtype=None):
        rows_output = rows.run(input.view(256, 4096))
        return partials.run(rows_output).view(())

    return wrapper
