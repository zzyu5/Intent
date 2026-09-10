import intent
import intent.language as I


@intent.kernel
def softmax_kernel(
    input: I.In[I.f32, ("M", "N")],
    output: I.Out[I.f32, ("M", "N")],
):
    rows = I.domain(0, input.shape[0])
    columns = I.domain(0, input.shape[1])

    for row in I.parallel(rows):
        values = input[row, columns]
        row_max = I.reduce.max(values, axis=0)
        shifted = (values - row_max) * 1.4426950408889634
        exponentials = I.exp2(shifted)
        normalizer = I.reduce.sum(exponentials, axis=0)
        output[row, columns] = I.fdiv(exponentials, normalizer)


def build(context):
    compiled = context.compile("softmax", softmax_kernel)

    def wrapper(input, dim, dtype=None):
        return compiled.run(input)

    return wrapper
