import torch
import intent
import intent.language as I


@intent.kernel
def fused_index_select_eq_kernel(
    input: I.In[I.f32, ("N", "M")],
    index: I.In[I.i64, ("K",)],
    other: I.f32,
    output: I.Out[I.i8, ("K", "M")],
):
    rows = I.domain(0, index.shape[0])
    columns = I.domain(0, input.shape[1])
    output[rows, columns] = I.cast(input[index[rows], columns] == other, I.i8)


def build(context):
    compiled = context.compile("fused_index_select_eq", fused_index_select_eq_kernel)

    def wrapper(input, dim, index, other, *, out=None):
        if out is None:
            return compiled.run(input, index, other).view(torch.bool)
        compiled(input, index, other, out.view(torch.int8))
        return out

    return wrapper
