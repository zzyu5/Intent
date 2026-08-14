import intent
import intent.language as I


ROWS = 8192
COLUMNS = 8192
NONZEROS_PER_ROW = 32
NONZEROS = ROWS * NONZEROS_PER_ROW
FEATURES = 128


@intent.kernel
def csr_spmm(
    row_offsets: I.In[I.i32, (ROWS + 1,)],
    column_indices: I.In[I.i32, (NONZEROS,)],
    values: I.In[I.f32, (NONZEROS,)],
    dense: I.In[I.f32, (COLUMNS, FEATURES)],
    output: I.Out[I.f32, (ROWS, FEATURES)],
):
    features = I.domain(0, FEATURES)
    for row in I.parallel(I.domain(0, ROWS)):
        start = row_offsets[row]
        stop = row_offsets[row + 1]
        for feature_region in I.parallel(
            I.partition(features, extent=I.auto("N_TILE"))
        ):
            accumulation = I.zeros((feature_region,), dtype=I.f32)
            for nonzero in range(start, stop):
                I.assume_in_bounds(nonzero, column_indices, axis=0)
                I.assume_in_bounds(nonzero, values, axis=0)
                column = column_indices[nonzero]
                I.assume_in_bounds(column, dense, axis=0)
                accumulation = accumulation + (
                    values[nonzero] * dense[column, feature_region]
                )
            output[row, feature_region] = accumulation
