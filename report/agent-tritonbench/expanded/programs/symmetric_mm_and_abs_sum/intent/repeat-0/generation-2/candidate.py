import intent
import intent.language as I


@intent.kernel
def symmetric_mm_and_abs_sum_kernel(
    A: I.In[I.f32, ("N", "M")],
    C: I.In[I.f32, ("N", "N")],
    alpha: I.f32,
    beta: I.f32,
    output: I.Out[I.f32, (1,)],
):
    product = I.matmul(A, A, acc_dtype=I.f32, transpose_rhs=True)
    combined = alpha * product + beta * C
    output[0] = I.reduce.sum(I.abs(combined), axis=(0, 1))


def build(context):
    compiled = context.compile(
        "symmetric_mm_and_abs_sum_kernel",
        symmetric_mm_and_abs_sum_kernel,
    )

    def wrapper(A, C, alpha, beta):
        return compiled.run(A, C, alpha, beta).reshape(())

    return wrapper
