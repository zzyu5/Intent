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
    result = I.matmul(lhs, rhs, acc_dtype=I.f32)
    output[rows, columns] = I.cast(result[rows, columns], I.f16)


def build(context):
    compiled = context.compile("tensordot", tensordot_kernel)

    def wrapper(a, b, dims):
        return compiled.run(a, b)

    return wrapper
