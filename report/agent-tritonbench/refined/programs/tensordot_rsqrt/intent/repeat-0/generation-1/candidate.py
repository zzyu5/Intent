import intent
import intent.language as I


@intent.kernel
def tensordot_rsqrt_kernel(
    a: I.In[I.f32, ("M", "K")],
    b: I.In[I.f32, ("K", "N")],
    output: I.Out[I.f32, ("M", "N")],
):
    rows = I.domain(0, a.shape[0])
    columns = I.domain(0, b.shape[1])
    product = I.matmul(a, b, acc_dtype=I.f32)
    output[rows, columns] = I.rsqrt(product[rows, columns])


def build(context):
    compiled = context.compile("tensordot_rsqrt", tensordot_rsqrt_kernel)

    def wrapper(a, b, dims):
        return compiled.run(a, b)

    return wrapper
