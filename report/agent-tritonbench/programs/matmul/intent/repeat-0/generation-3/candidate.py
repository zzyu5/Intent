import torch
import intent
import intent.language as I


@intent.kernel
def matrix_product(
    input: I.In[I.f16, ("M", "K")],
    other: I.In[I.f16, ("K", "N")],
    output: I.Out[I.f16, ("M", "N")],
):
    product = I.matmul(input, other, acc_dtype=I.f32)
    rows = I.domain(0, product.shape[0])
    columns = I.domain(0, product.shape[1])
    output[rows, columns] = I.cast(product[rows, columns], I.f16)


def build(context):
    compiled = context.compile("matrix_product", matrix_product)

    def wrapper(input, other, *, out=None):
        if out is None:
            return compiled.run(input, other)
        compiled(input, other, out)
        return out

    return wrapper
