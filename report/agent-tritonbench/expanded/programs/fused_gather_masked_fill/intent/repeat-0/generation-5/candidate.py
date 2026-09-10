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
    columns = I.domain(0, input.shape[1])
    fill = I.full(input.shape, value, dtype=I.f32)
    for row in I.parallel(I.domain(0, input.shape[0])):
        gathered = input[index[row, columns], columns]
        output[row, columns] = fill[row, columns] if mask[row, columns] else gathered


def build(context):
    compiled = context.compile(
        "fused_gather_masked_fill_kernel",
        fused_gather_masked_fill_kernel,
    )

    def wrapper(input, dim, index, mask, value, *, sparse_grad=False, out=None):
        if out is None:
            return compiled.run(input, index, mask, value)
        compiled(input, index, mask, value, out)
        return out

    return wrapper
