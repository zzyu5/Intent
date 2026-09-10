import intent
import intent.language as I
import torch


@intent.kernel
def materialize_absolute_matrix(
    A: I.In[I.f32, ("N", "M")],
    AT: I.In[I.f32, ("M", "N")],
    C: I.In[I.f32, ("N", "N")],
    alpha: I.f32,
    beta: I.f32,
    output: I.Out[I.f32, ("N", "N")],
):
    product = I.matmul(A, AT, acc_dtype=I.f32)
    result = alpha * product + beta * C
    absolute = I.maximum(result, -result)

    rows = I.domain(0, output.shape[0])
    columns = I.domain(0, output.shape[1])
    output[rows, columns] = absolute


@intent.kernel
def reduce_absolute_matrix(
    values: I.In[I.f32, ("N", "N")],
    output: I.Out[I.f32, ()],
):
    total = I.reduce.sum(values, axis=(0, 1), acc_dtype=I.f32)
    output[()] = total


def build(context):
    materialize = context.compile(
        "symmetric_mm_materialize_absolute", materialize_absolute_matrix
    )
    reduce = context.compile("symmetric_mm_reduce_absolute", reduce_absolute_matrix)

    def wrapper(A, C, alpha, beta):
        absolute_matrix = torch.empty_like(C)
        materialize(A, A.transpose(0, 1), C, alpha, beta, absolute_matrix)
        return reduce.run(absolute_matrix)

    return wrapper
