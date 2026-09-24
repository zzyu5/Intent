import torch
import intent
import intent.language as I


@intent.kernel
def matmul_values(
    lhs: I.In[I.f16, (1024, 1024)],
    rhs: I.In[I.f16, (1024, 1024)],
    out: I.Out[I.f16, (1024, 1024)],
):
    rows = I.domain(0, 1024)
    columns = I.domain(0, 1024)
    reduction = I.domain(0, 1024)
    product = I.matmul(lhs[rows, reduction], rhs[reduction, columns], acc_dtype=I.f32)
    out[rows, columns] = I.cast(product, I.f16)


def build(context):
    artifact = context.compile("matmul_values", matmul_values)

    def matmul(tensor1, tensor2):
        result = torch.empty((1024, 1024), device=tensor1.device, dtype=tensor1.dtype)
        artifact(tensor1, tensor2, result)
        return result

    return matmul
