import intent
import intent.language as I


@intent.kernel
def softmax_kernel(
    input: I.In[I.f32, ("M", "N")],
    output: I.Out[I.f32, ("M", "N")],
):
    rows = I.domain(0, input.shape[0])
    columns = I.domain(0, input.shape[1])
    values = input[rows, columns]

    row_max = I.reduce.max(values, axis=1)
    row_max = I.reshape(row_max, (input.shape[0], 1))
    shifted = values - row_max

    log2_e = I.cast(1.4426950408889634, I.f32)
    weights = I.exp2(shifted * log2_e)
    row_sum = I.reduce.sum(weights, axis=1)
    row_sum = I.reshape(row_sum, (input.shape[0], 1))

    output[rows, columns] = weights / row_sum


def build(context):
    compiled = context.compile("softmax_kernel", softmax_kernel)

    def wrapper(input, dim, dtype=None):
        return compiled.run(input)

    return wrapper
