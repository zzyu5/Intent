import intent
import intent.language as I


ROWS = 4093
COLUMNS = 8191


@intent.kernel
def matrix_transpose(
    x: I.In[I.f16, ("M", "N")],
    output: I.Out[I.f16, ("N", "M")],
):
    M, N = x.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    for row_tile in I.parallel(I.partition(rows, extent=I.auto("M_TILE"))):
        for column_tile in I.parallel(
            I.partition(columns, extent=I.auto("N_TILE"))
        ):
            tile = x[row_tile, column_tile]
            output[column_tile, row_tile] = I.transpose(
                tile, permutation=(1, 0)
            )
