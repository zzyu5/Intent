import torch
import intent
import intent.language as I


@intent.kernel
def softmax_rows(
    x: I.In[I.f32, ("M", "N")],
    out: I.Out[I.f32, ("M", "N")],
):
    M, N = x.shape
    rows = I.domain(0, M)
    cols = I.domain(0, N)

    for row in I.parallel(rows):
        row_values = x[row, cols]
        row_max = I.reduce.max(row_values, axis=0)
        shifted = row_values - row_max
        exponentials = I.exp(shifted)
        denominator = I.reduce.sum(exponentials, axis=0)
        out[row, cols] = exponentials / denominator


def build(context):
    compiled = context.compile("softmax_rows", softmax_rows)

    def softmax(input, dim, dtype=None):
        return compiled.run(input)

    return softmax
