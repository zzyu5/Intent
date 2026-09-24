import torch
import intent
import intent.language as I


@intent.kernel
def matmul_kernel(
    tensor1: I.In[I.f16, ("M", "K")],
    tensor2: I.In[I.f16, ("K", "N")],
    output: I.Out[I.f16, ("M", "N")],
):
    rows = I.domain(0, tensor1.shape[0])
    columns = I.domain(0, tensor2.shape[1])
    product = I.matmul(tensor1, tensor2, acc_dtype=I.f32)
    output[rows, columns] = I.cast(product, I.f16)


def build(context):
    compiled_matmul = context.compile("matmul_f16_1024", matmul_kernel)

    def matmul(tensor1, tensor2):
        output = torch.empty(
            (tensor1.shape[0], tensor2.shape[1]),
            dtype=tensor1.dtype,
            device=tensor1.device,
        )
        compiled_matmul(tensor1, tensor2, output)
        return output

    return matmul
