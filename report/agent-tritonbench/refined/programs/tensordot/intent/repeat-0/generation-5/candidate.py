import intent
import intent.language as I


@intent.kernel
def tensordot_kernel(
    a: I.In[I.f16, (1024, 8, 8)],
    b: I.In[I.f16, (8, 8, 1024)],
    output: I.Out[I.f16, (1024, 1024)],
):
    lhs = I.reshape(a, (1024, 64))
    rhs = I.reshape(b, (64, 1024))

    rows = I.domain(0, 1024)
    columns = I.domain(0, 1024)
    reduction = I.domain(0, 64)
    row_parts = I.domain(0, 8)
    column_parts = I.domain(0, 8)

    for row_part in I.parallel(row_parts):
        row_region = rows[row_part * 128 : (row_part + 1) * 128]
        for column_part in I.parallel(column_parts):
            column_region = columns[column_part * 128 : (column_part + 1) * 128]
            tile = I.matmul(
                lhs[row_region, reduction],
                rhs[reduction, column_region],
                acc_dtype=I.f32,
            )
            output[row_region, column_region] = I.cast(tile, I.f16)


def build(context):
    compiled = context.compile("tensordot_tiled", tensordot_kernel)

    def wrapper(a, b, dims):
        return compiled.run(a, b)

    return wrapper
