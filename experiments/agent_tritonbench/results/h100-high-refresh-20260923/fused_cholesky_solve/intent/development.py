import torch
import intent
import intent.language as I


@intent.kernel
def cholesky_factor(A: I.In[I.f32, (256, 256)], L: I.Out[I.f32, (256, 256)]):
    for column in I.domain(0, 256):
        prefix = I.domain(0, column)
        previous = L[column, prefix]
        pivot = I.sqrt(A[column, column] - I.reduce.sum(previous * previous, axis=0))
        L[column, column] = pivot
        for row in I.parallel(I.domain(column + 1, 256)):
            correction = I.reduce.sum(L[row, prefix] * previous, axis=0)
            L[row, column] = (A[row, column] - correction) / pivot


@intent.kernel
def cholesky_substitute(
    L: I.In[I.f32, (256, 256)],
    b: I.In[I.f32, (256, 1)],
    x: I.Out[I.f32, (256, 1)],
):
    rows = I.domain(0, 256)
    indices = I.indices(rows)
    residual = b[rows, 0]
    for row in I.domain(0, 256):
        value = residual[row] / L[row, row]
        below = indices > row
        column = I.gather(L, (indices, row), valid=below, fill=0.0)
        residual = I.select(below, residual - column * value,
                            I.select(indices == row, value, residual))
    for ordinal in I.domain(0, 256):
        row = 255 - ordinal
        value = residual[row] / L[row, row]
        above = indices < row
        column = I.gather(L, (row, indices), valid=above, fill=0.0)
        residual = I.select(above, residual - column * value,
                            I.select(indices == row, value, residual))
    x[rows, 0] = residual


def build(context):
    factor = context.compile("fused_cholesky_factor", cholesky_factor)
    substitute = context.compile("fused_cholesky_substitute", cholesky_substitute)

    def fused_cholesky_solve(A, b):
        L = torch.empty_like(A)
        x = torch.empty_like(b)
        factor(A, L)
        substitute(L, b, x)
        return x

    return fused_cholesky_solve
