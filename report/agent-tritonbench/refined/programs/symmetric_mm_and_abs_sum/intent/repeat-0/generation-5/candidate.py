import intent
import intent.language as I
import torch


@intent.kernel
def symmetric_product(
    A: I.In[I.f32, ("N", "M")],
    AT: I.In[I.f32, ("M", "N")],
    output: I.Out[I.f32, ("N", "N")],
):
    product = I.matmul(A, AT, acc_dtype=I.f32)
    rows = I.domain(0, output.shape[0])
    columns = I.domain(0, output.shape[1])
    output[rows, columns] = product


@intent.kernel
def scaled_absolute(
    product: I.In[I.f32, ("N", "N")],
    C: I.In[I.f32, ("N", "N")],
    alpha: I.f32,
    beta: I.f32,
    output: I.Out[I.f32, ("N", "N")],
):
    result = alpha * product + beta * C
    absolute = I.maximum(result, -result)
    rows = I.domain(0, output.shape[0])
    columns = I.domain(0, output.shape[1])
    output[rows, columns] = absolute


@intent.kernel
def sum_absolute(
    values: I.In[I.f32, ("N", "N")],
    output: I.Out[I.f32, ()],
):
    total = I.reduce.sum(values, axis=(0, 1), acc_dtype=I.f32)
    output[()] = total


def build(context):
    product_kernel = context.compile("symmetric_mm_product", symmetric_product)
    absolute_kernel = context.compile(
        "symmetric_mm_scaled_absolute", scaled_absolute
    )
    sum_kernel = context.compile("symmetric_mm_sum_absolute", sum_absolute)

    def wrapper(A, C, alpha, beta):
        product = torch.empty_like(C)
        absolute = torch.empty_like(C)
        output = torch.empty((), dtype=C.dtype, device=C.device)
        product_kernel(A, A.transpose(0, 1), product)
        absolute_kernel(product, C, alpha, beta, absolute)
        sum_kernel(absolute, output)
        return output

    return wrapper
