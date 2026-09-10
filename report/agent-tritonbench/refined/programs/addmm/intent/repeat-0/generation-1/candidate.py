import intent
import intent.language as I


@intent.kernel
def addmm_kernel(
    input: I.In[I.f16, ("M", "N")],
    mat1: I.In[I.f16, ("M", "K")],
    mat2: I.In[I.f16, ("K", "N")],
    beta: I.f32,
    alpha: I.f32,
    output: I.Out[I.f16, ("M", "N")],
):
    rows = I.domain(0, input.shape[0])
    columns = I.domain(0, input.shape[1])

    product = I.matmul(mat1, mat2, acc_dtype=I.f32)
    input_f32 = I.cast(input[rows, columns], I.f32)
    result = beta * input_f32 + alpha * product
    output[rows, columns] = I.cast(result, I.f16)


def build(context):
    compiled = context.compile("addmm_kernel", addmm_kernel)

    def wrapper(input, mat1, mat2, *, beta=1, alpha=1, out=None):
        if out is None:
            return compiled.run(input, mat1, mat2, beta, alpha)

        compiled(input, mat1, mat2, beta, alpha, out)
        return out

    return wrapper
