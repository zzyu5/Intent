import intent
import intent.language as I


@intent.kernel
def symmetric_mm_and_abs_sum_rows(
    A: I.In[I.f32, ("N", "M")],
    C: I.In[I.f32, ("N", "N")],
    alpha: I.f32,
    beta: I.f32,
    output: I.Out[I.f32, ("N",)],
):
    product = I.matmul(A, A, acc_dtype=I.f32, transpose_rhs=True)
    alpha_tensor = I.full(product.shape, alpha, dtype=I.f32)
    beta_tensor = I.full(C.shape, beta, dtype=I.f32)
    combined = alpha_tensor * product + beta_tensor * C
    row_sums = I.reduce.sum(I.abs(combined), axis=1, acc_dtype=I.f32)
    rows = I.domain(0, output.shape[0])
    output[rows] = row_sums[rows]


@intent.kernel
def sum_rows(
    rows: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, (1,)],
):
    total = I.cast(0.0, I.f32)
    for index in I.domain(0, rows.shape[0]):
        total = total + rows[index]
    output[0] = total


def build(context):
    rows_kernel = context.compile(
        "symmetric_mm_and_abs_sum_rows",
        symmetric_mm_and_abs_sum_rows,
    )
    sum_kernel = context.compile("sum_rows", sum_rows)

    def wrapper(A, C, alpha, beta):
        rows = rows_kernel.run(A, C, alpha, beta)
        return sum_kernel.run(rows).reshape(())

    return wrapper
