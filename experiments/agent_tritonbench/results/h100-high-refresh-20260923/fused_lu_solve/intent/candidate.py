import torch
import intent
import intent.language as I


N = 256


@intent.kernel
def lu_factor(
    A: I.In[I.f32, (256, 256)],
    LU: I.Out[I.f32, (256, 256)],
):
    # The outer panel loop is ordered; rows below the pivot are independent.
    lu = I.buffer((256, 256), I.f32, init=A)
    for k in I.domain(0, 256):
        pivot = k
        pivot_abs = I.abs(lu[k, k])
        for i in I.domain(k + 1, 256):
            candidate_abs = I.abs(lu[i, k])
            if candidate_abs > pivot_abs:
                pivot = i
                pivot_abs = candidate_abs

        if pivot != k:
            for j in I.domain(0, 256):
                tmp = lu[k, j]
                lu[k, j] = lu[pivot, j]
                lu[pivot, j] = tmp

        diagonal = lu[k, k]
        for i in I.parallel(I.domain(k + 1, 256)):
            multiplier = lu[i, k] / diagonal
            lu[i, k] = multiplier
            for j in I.domain(k + 1, 256):
                lu[i, j] = lu[i, j] - multiplier * lu[k, j]

    rows = I.domain(0, 256)
    columns = I.domain(0, 256)
    LU[rows, columns] = lu[rows, columns]


@intent.kernel
def triangular_solve(
    LU: I.In[I.f32, (256, 256)],
    b: I.In[I.f32, (256,)],
    out: I.Out[I.f32, (256,)],
):
    # Forward and backward substitutions are strict recurrences.
    rhs = I.buffer((256,), I.f32, init=b)
    x = I.buffer((256,), I.f32, init=b)

    for i in I.domain(0, 256):
        value = rhs[i]
        for j in I.domain(0, i):
            value = value - LU[i, j] * rhs[j]
        rhs[i] = value

    for ordinal in I.domain(0, 256):
        i = 256 - 1 - ordinal
        value = rhs[i]
        for j in I.domain(i + 1, 256):
            value = value - LU[i, j] * x[j]
        x[i] = value / LU[i, i]

    indices = I.domain(0, 256)
    out[indices] = x[indices]


def build(context):
    factor_artifact = context.compile("fused_lu_factor", lu_factor)
    solve_artifact = context.compile("fused_lu_triangular_solve", triangular_solve)

    def fused_lu_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        lu = torch.empty((N, N), dtype=A.dtype, device=A.device)
        out = torch.empty_like(b)
        factor_artifact(A, lu)
        solve_artifact(lu, b, out)
        return out

    return fused_lu_solve
