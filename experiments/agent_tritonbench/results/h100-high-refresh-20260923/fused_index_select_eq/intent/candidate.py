import torch
import intent
import intent.language as I


@intent.kernel
def fused_index_select_eq_kernel(
    input: I.In[I.f32, ("M", "N")],
    index: I.In[I.i64, ("K",)],
    other: I.f32,
    output: I.Out[I.bool, ("K", "N")],
):
    M, N = input.shape
    K = index.shape[0]
    rows = I.domain(0, K)
    columns = I.domain(0, N)

    selected_rows = I.cast(index[rows], I.index)
    selected = input[selected_rows, columns]
    output[rows, columns] = selected == other


def build(context):
    kernel = context.compile("fused_index_select_eq", fused_index_select_eq_kernel)

    def fused_index_select_eq(input, dim, index, other, *, out=None):
        if out is None:
            return kernel.run(input, index, other)
        kernel(input, index, other, out)
        return out

    return fused_index_select_eq
