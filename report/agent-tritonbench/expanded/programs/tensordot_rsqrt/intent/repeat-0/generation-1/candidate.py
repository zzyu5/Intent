import intent
import intent.language as I


@intent.kernel
def tensordot_rsqrt(
    a: I.In[I.f32, ("M", "K")],
    b: I.In[I.f32, ("K", "N")],
    output: I.Out[I.f32, ("M", "N")],
):
    product = I.matmul(a, b, acc_dtype=I.f32)
    rows = I.domain(0, a.shape[0])
    columns = I.domain(0, b.shape[1])
    output[rows, columns] = I.rsqrt(product[rows, columns])


def build(context):
    compiled = context.compile("tensordot_rsqrt", tensordot_rsqrt)

    def wrapper(a, b, dims):
        return compiled.run(a, b)

    return wrapper
