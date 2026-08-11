import intent
import intent.language as I


BATCH = 512
ROWS = 17
COLUMNS = 31


@intent.kernel
def ordered_product_prefix(
    x: I.In[I.f32, ("B", "M", "N")],
    output: I.Out[I.f32, ("B", "M", "N")],
):
    B, M, N = x.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    for batch in I.parallel(I.domain(0, B)):
        prefix = I.cast(0.0, I.f32)
        for row, column in I.ordered((rows, columns)):
            prefix = prefix + x[batch, row, column]
            output[batch, row, column] = prefix
