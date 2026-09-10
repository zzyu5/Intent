import intent
import intent.language as I


@intent.kernel
def symmetric_mm_and_abs_sum(
    A: I.In[I.f32, ("N", "M")],
    C: I.In[I.f32, ("N", "N")],
    alpha: I.f32,
    beta: I.f32,
    output: I.Out[I.f32, ()],
):
    product = I.matmul(A, A, transpose_rhs=True, acc_dtype=I.f32)
    result = alpha * product + beta * C
    absolute = I.maximum(result, -result)
    total = I.reduce.sum(absolute, axis=(0, 1), acc_dtype=I.f32)
    output[()] = total


def build(context):
    compiled = context.compile("symmetric_mm_and_abs_sum", symmetric_mm_and_abs_sum)

    def wrapper(A, C, alpha, beta):
        return compiled.run(A, C, alpha, beta)

    return wrapper
