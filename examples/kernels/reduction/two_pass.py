import intent
import intent.language as I


ROWS = 128
COLUMNS = 257
PARTS = 300


@intent.kernel
def partitioned_max_partial(
    x: I.In[I.f32, ("M", "N")],
    partial: I.Out[I.f32, ("M", "P")],
):
    M, N = x.shape
    _, P = partial.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        for part, region in I.parallel(I.partition(columns, count=P)):
            partial[row, part] = I.reduce.max(
                x[row, region], axis=0, identity=-I.inf
            )


@intent.kernel
def partitioned_max_reduce(
    partial: I.In[I.f32, ("M", "P")],
    output: I.Out[I.f32, ("M",)],
):
    M, P = partial.shape
    parts = I.domain(0, P)
    for row in I.parallel(I.domain(0, M)):
        output[row] = I.reduce.max(
            partial[row, parts], axis=0, identity=-I.inf
        )
