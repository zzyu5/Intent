import intent
import intent.language as I


@intent.kernel
def matrix_multiply(
    input: I.In[I.f16, ("M", "K")],
    other: I.In[I.f16, ("K", "N")],
    output: I.Out[I.f16, ("M", "N")],
):
    rows = I.domain(0, input.shape[0])
    columns = I.domain(0, other.shape[1])
    product = I.matmul(input, other, acc_dtype=I.f32)
    output[rows, columns] = I.cast(product, I.f16)


def build(context):
    compiled = context.compile("matrix_multiply", matrix_multiply, constexprs={})

    def wrapper(input, other, *, out=None):
        if out is None:
            return compiled.run(input, other)
        compiled(input, other, out)
        return out

    return wrapper
