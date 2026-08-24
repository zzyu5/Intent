import intent
import intent.language as I


@intent.kernel
def split_k_partial(
    a: I.In[I.f16, ("M", "K")],
    b: I.In[I.f16, ("K", "N")],
    partial: I.Out[I.f32, ("P", "M", "N")],
):
    P, M, N = partial.shape
    K = a.shape[1]
    parts = I.domain(0, P)
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    reduction_axis = I.domain(0, K)
    width = (K + P - 1) // P

    for part in I.parallel(parts):
        begin = I.minimum(part * width, K)
        end = I.minimum((part + 1) * width, K)
        reduction = reduction_axis[begin:end]
        partial[part, rows, columns] = I.contract(
            a[rows, reduction],
            b[reduction, columns],
            reduce=((1, 0),),
            acc_dtype=I.f32,
        )


@intent.kernel
def split_k_combine(
    partial: I.In[I.f32, ("P", "M", "N")],
    output: I.Out[I.f16, ("M", "N")],
):
    P, M, N = partial.shape
    parts = I.domain(0, P)
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    output[rows, columns] = I.cast(
        I.reduce.sum(partial[parts, rows, columns], axis=0, identity=0.0),
        I.f16,
    )


def run_split_k(a, b, parts):
    """Host pseudocode: allocation and launch order are author-visible."""
    partial = allocate_tensor((parts, a.shape[0], b.shape[1]), dtype="float32")
    output = allocate_tensor((a.shape[0], b.shape[1]), dtype="float16")
    launch(split_k_partial, a, b, partial)
    launch(split_k_combine, partial, output)
    return output
