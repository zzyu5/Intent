import intent
import intent.language as I


@intent.kernel
def fused_gather_masked_fill_kernel(
    input: I.In[I.f32, ("M", "N")],
    index: I.In[I.i64, ("M", "N")],
    mask: I.In[I.bool, ("M", "N")],
    value: I.f32,
    output: I.Out[I.f32, ("M", "N")],
):
    rows = I.domain(0, input.shape[0])
    columns = I.domain(0, input.shape[1])

    for row in I.parallel(rows):
        for column in I.parallel(columns):
            gathered = input[index[row, column], column]
            output[row, column] = value if mask[row, column] else gathered


def build(context):
    compiled = context.compile(
        "fused_gather_masked_fill",
        fused_gather_masked_fill_kernel,
    )

    def wrapper(input, dim, index, mask, value, *, sparse_grad=False, out=None):
        if out is None:
            return compiled.run(input, index, mask, value)
        compiled(input, index, mask, value, out)
        return out

    return wrapper
