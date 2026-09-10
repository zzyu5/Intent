import intent
import intent.language as I


@intent.kernel
def tensordot_kernel(
    a: I.In[I.f16, ("M", "K0", "K1")],
    b: I.In[I.f16, ("K0", "K1", "N")],
    output: I.Out[I.f16, ("M", "N")],
):
    rows = I.domain(0, a.shape[0])
    columns = I.domain(0, b.shape[2])
    result = I.contract(
        a,
        b,
        reduce=((1, 0), (2, 1)),
        batch=(),
        acc_dtype=I.f32,
    )
    output[rows, columns] = I.cast(result[rows, columns], I.f16)


def build(context):
    compiled = context.compile("tensordot", tensordot_kernel)

    def wrapper(a, b, dims):
        return compiled.run(a, b)

    return wrapper
