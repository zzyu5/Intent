import torch
import intent
import intent.language as I


@intent.kernel
def least_squares(
    A: I.In[I.f32, (64, 16)],
    b: I.In[I.f32, (64, 1)],
    out: I.Out[I.f32, (16, 1)],
):
    rows = I.domain(0, 64)
    columns = I.domain(0, 16)
    row = I.indices(rows)
    column = I.indices(columns)
    matrix = A[rows, columns]
    rhs = b[rows, 0]
    zero = I.cast(0.0, I.f32)

    for j in range(16):
        selected_column = I.reduce.sum(
            I.select(I.reshape(column == j, (1, 16)), matrix, zero),
            axis=1, acc_dtype=I.f32)
        vector = I.select(row >= j, selected_column, zero)
        leading = I.reduce.sum(I.select(row == j, vector, zero),
                                axis=0, acc_dtype=I.f32)
        norm = I.sqrt(I.reduce.sum(vector * vector, axis=0, acc_dtype=I.f32))
        diagonal = I.select(leading >= zero, -norm, norm)
        vector = I.select(row == j, vector - diagonal, vector)
        squared_norm = I.reduce.sum(vector * vector, axis=0, acc_dtype=I.f32)
        scale = I.select(squared_norm > zero,
                         I.cast(2.0, I.f32) / squared_norm, zero)
        projections = scale * I.reduce.sum(
            I.reshape(vector, (64, 1)) * matrix, axis=0, acc_dtype=I.f32)
        active = I.reshape(row >= j, (64, 1)) & I.reshape(column >= j, (1, 16))
        matrix = I.select(active, matrix - I.outer(vector, projections), matrix)
        rhs_projection = scale * I.reduce.sum(vector * rhs, axis=0, acc_dtype=I.f32)
        rhs = rhs - vector * rhs_projection

    solution = I.full((16,), 0.0, dtype=I.f32)
    for reverse_i in range(16):
        i = 15 - reverse_i
        coefficients = I.reduce.sum(
            I.select(I.reshape(row == i, (64, 1)), matrix, zero),
            axis=0, acc_dtype=I.f32)
        remainder = I.reduce.sum(
            I.select(column > i, coefficients * solution, zero),
            axis=0, acc_dtype=I.f32)
        diagonal = I.reduce.sum(I.select(column == i, coefficients, zero),
                                axis=0, acc_dtype=I.f32)
        value = I.reduce.sum(I.select(row == i, rhs, zero),
                             axis=0, acc_dtype=I.f32)
        solution = I.select(column == i, (value - remainder) / diagonal, solution)
    out[columns, 0] = solution


def build(context):
    kernel = context.compile("least_squares", least_squares)

    def least_squares_qr(A, b, mode="reduced", out=None):
        if mode != "reduced":
            raise NotImplementedError("development program uses the reduced profile")
        if out is None:
            out = torch.empty((16, 1), device=A.device, dtype=A.dtype)
        kernel(A, b, out)
        return out

    return least_squares_qr
