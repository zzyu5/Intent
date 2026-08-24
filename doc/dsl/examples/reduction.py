import intent
import intent.language as I


@intent.fn
def maximum_pair(lhs, rhs):
    return I.maximum(lhs, rhs)


@intent.kernel
def row_max(
    x: I.In[I.f32, ("M", "N")],
    output: I.Out[I.f32, ("M",)],
):
    M, N = x.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)

    output[rows] = I.reduce(
        x[rows, columns],
        axes=(1,),
        identity=-I.inf,
        combine=maximum_pair,
    )
