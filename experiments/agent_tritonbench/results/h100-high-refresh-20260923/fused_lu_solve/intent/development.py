import torch
import intent
import intent.language as I


N = 256


@intent.kernel
def lu_factor(A: I.In[I.f32, (N, N)], LU: I.Out[I.f32, (N, N)]):
    lu = I.buffer((N, N), I.f32, init=A)
    for k in I.domain(0, N):
        pivot = k
        pivot_abs = I.abs(lu[k, k])
        for i in I.domain(k + 1, N):
            candidate_abs = I.abs(lu[i, k])
            if candidate_abs > pivot_abs:
                pivot = i
                pivot_abs = candidate_abs
        if pivot != k:
            for j in I.domain(0, N):
                previous = lu[k, j]
                lu[k, j] = lu[pivot, j]
                lu[pivot, j] = previous
        diagonal = lu[k, k]
        for i in I.parallel(I.domain(k + 1, N)):
            multiplier = lu[i, k] / diagonal
            lu[i, k] = multiplier
            for j in I.domain(k + 1, N):
                lu[i, j] = lu[i, j] - multiplier * lu[k, j]
    rows = I.domain(0, N)
    columns = I.domain(0, N)
    LU[rows, columns] = lu[rows, columns]


@intent.kernel
def triangular_solve(
    LU: I.In[I.f32, (N, N)],
    b: I.In[I.f32, (N,)],
    out: I.Out[I.f32, (N,)],
):
    rows = I.domain(0, N)
    indices = I.indices(rows)
    residual = b[rows]
    for column in I.domain(0, N):
        solved = residual[column]
        residual = I.select(indices > column, residual - LU[rows, column] * solved, residual)
    for ordinal in I.domain(0, N):
        column = N - 1 - ordinal
        solved = residual[column] / LU[column, column]
        residual = I.select(indices < column, residual - LU[rows, column] * solved, residual)
        residual = I.select(indices == column, solved, residual)
    out[rows] = residual


def build(context):
    factor = context.compile("fused_lu_factor", lu_factor)
    substitute = context.compile("fused_lu_triangular_solve", triangular_solve)

    def fused_lu_solve(A, b):
        lu = torch.empty((N, N), dtype=A.dtype, device=A.device)
        output = torch.empty_like(b)
        factor(A, lu)
        substitute(lu, b, output)
        return output

    return fused_lu_solve
