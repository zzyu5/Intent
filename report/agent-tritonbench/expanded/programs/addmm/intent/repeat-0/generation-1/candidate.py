import intent
import intent.language as I


@intent.kernel
def addmm_kernel(
    input: I.In[I.f16, ("M", "N")],
    mat1: I.In[I.f16, ("M", "K")],
    mat2: I.In[I.f16, ("K", "N")],
    output: I.Out[I.f16, ("M", "N")],
):
    product = I.matmul(mat1, mat2, acc_dtype=I.f32)
    rows = I.domain(0, output.shape[0])
    columns = I.domain(0, output.shape[1])
    value = I.cast(input[rows, columns], I.f32) + product[rows, columns]
    output[rows, columns] = I.cast(value, I.f16)


def build(context):
    compiled = context.compile("addmm_kernel", addmm_kernel)

    def wrapper(input, mat1, mat2, *, beta=1, alpha=1, out=None):
        if out is None:
            return compiled.run(input, mat1, mat2)
        compiled(input, mat1, mat2, out)
        return out

    return wrapper
