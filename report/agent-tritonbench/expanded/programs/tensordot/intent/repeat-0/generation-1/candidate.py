import intent
import intent.language as I


@intent.kernel
def tensordot_kernel(
    a: I.In[I.f16, ("M", "K1", "K2")],
    b: I.In[I.f16, ("K1", "K2", "N")],
    output: I.Out[I.f16, ("M", "N")],
):
    m = I.domain(0, a.shape[0])
    k1 = I.domain(0, a.shape[1])
    k2 = I.domain(0, a.shape[2])
    n = I.domain(0, b.shape[2])

    result = I.contract(
        a[m, k1, k2],
        b[k1, k2, n],
        reduce=((1, 0), (2, 1)),
        batch=(),
        acc_dtype=I.f32,
    )
    output[m, n] = I.cast(result, I.f16)


def build(context):
    compiled = context.compile("tensordot_kernel", tensordot_kernel)

    def wrapper(a, b, dims):
        return compiled.run(a, b)

    return wrapper
