import torch
import intent
import intent.language as I


@intent.kernel
def ldl_factor(
    matrix: I.In[I.f32, ("N", "N")],
    factor: I.InOut[I.f32, ("N", "N")],
):
    n = matrix.shape[0]
    for row in I.domain(0, n):
        for column in I.domain(0, row + 1):
            value = matrix[row, column]
            for inner in I.domain(0, column):
                value = value - (
                    factor[row, inner]
                    * factor[inner, inner]
                    * factor[column, inner]
                )
            if row == column:
                factor[row, column] = value
            else:
                factor[row, column] = I.fdiv(value, factor[column, column])


@intent.kernel
def ldl_forward(
    factor: I.In[I.f32, ("N", "N")],
    rhs: I.In[I.f32, ("N", "K")],
    intermediate: I.Out[I.f32, ("N", "K")],
):
    n = rhs.shape[0]
    columns = rhs.shape[1]
    for column in I.domain(0, columns):
        for row in I.domain(0, n):
            value = rhs[row, column]
            for inner in I.domain(0, row):
                value = value - factor[row, inner] * intermediate[inner, column]
            intermediate[row, column] = I.fdiv(value, factor[row, row])


@intent.kernel
def ldl_backward(
    factor: I.In[I.f32, ("N", "N")],
    intermediate: I.In[I.f32, ("N", "K")],
    output: I.Out[I.f32, ("N", "K")],
):
    n = intermediate.shape[0]
    columns = intermediate.shape[1]
    for column in I.domain(0, columns):
        for reverse_row in I.domain(0, n):
            row = n - 1 - reverse_row
            value = intermediate[row, column]
            for inner in I.domain(row + 1, n):
                value = value - factor[inner, row] * output[inner, column]
            output[row, column] = value


def build(context):
    factor_kernel = context.compile("ldl_factor", ldl_factor)
    forward_kernel = context.compile("ldl_forward", ldl_forward)
    backward_kernel = context.compile("ldl_backward", ldl_backward)

    def wrapper(A, b):
        factor = torch.empty_like(A)
        factor_kernel.run(A, factor)
        intermediate = forward_kernel.run(factor, b)
        return backward_kernel.run(factor, intermediate)

    return wrapper
