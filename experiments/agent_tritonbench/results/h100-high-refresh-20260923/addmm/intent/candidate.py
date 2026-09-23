import torch
import intent
import intent.language as I


@intent.kernel
def addmm_kernel(
    input: I.In[I.f16, ("M", "N")],
    mat1: I.In[I.f16, ("M", "K")],
    mat2: I.In[I.f16, ("K", "N")],
    beta: I.f32,
    alpha: I.f32,
    out: I.Out[I.f16, ("M", "N")],
):
    m, n = input.shape
    k = mat1.shape[1]
    rows = I.domain(0, m)
    columns = I.domain(0, n)
    reduction = I.domain(0, k)

    product = I.matmul(
        mat1[rows, reduction],
        mat2[reduction, columns],
        acc_dtype=I.f32,
    )

    if beta == I.cast(0.0, I.f32):
        value = alpha * product
    else:
        value = beta * I.cast(input[rows, columns], I.f32) + alpha * product
    out[rows, columns] = I.cast(value, I.f16)


def build(context):
    compiled = context.compile("addmm_kernel", addmm_kernel)

    def addmm(input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor,
              beta: float = 1, alpha: float = 1,
              out: torch.Tensor = None) -> torch.Tensor:
        result = out if out is not None else torch.empty_like(input)
        compiled(input, mat1, mat2, float(beta), float(alpha), result)
        return result

    return addmm
