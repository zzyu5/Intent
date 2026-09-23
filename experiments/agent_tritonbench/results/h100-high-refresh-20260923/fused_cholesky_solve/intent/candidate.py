import torch
import intent
import intent.language as I


N = 256
K = 1


@intent.kernel
def cholesky_kernel(
    A: I.In[I.f32, (N, N)],
    L: I.Out[I.f32, (N, N)],
):
    # Each completed column exposes the next pivot and all independent
    # updates below it.  The column loop is ordered; the trailing rows are
    # disjoint and therefore unordered.
    for column in I.domain(0, N):
        diagonal = A[column, column]
        for previous in I.domain(0, column):
            value = L[column, previous]
            diagonal = diagonal - value * value
        pivot = I.sqrt(diagonal)
        L[column, column] = pivot

        for row in I.parallel(I.domain(column + 1, N)):
            value = A[row, column]
            for previous in I.domain(0, column):
                value = value - L[row, previous] * L[column, previous]
            L[row, column] = value / pivot


@intent.kernel
def solve_kernel(
    L: I.In[I.f32, (N, N)],
    b: I.In[I.f32, (N, K)],
    x: I.Out[I.f32, (N, K)],
):
    # The right-hand side is one column in the fixed profile.  Keep the
    # forward solution in a kernel-local buffer, then substitute backwards.
    y = I.buffer((N, K), I.f32)

    for row in I.domain(0, N):
        value = b[row, 0]
        for previous in I.domain(0, row):
            value = value - L[row, previous] * y[previous, 0]
        y[row, 0] = value / L[row, row]

    for ordinal in I.domain(0, N):
        row = N - 1 - ordinal
        value = y[row, 0]
        for following in I.domain(row + 1, N):
            value = value - L[following, row] * x[following, 0]
        x[row, 0] = value / L[row, row]


def build(context):
    cholesky = context.compile("fused_cholesky_factor", cholesky_kernel)
    solve = context.compile("fused_cholesky_substitute", solve_kernel)

    def fused_cholesky_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        L = torch.empty((N, N), device=A.device, dtype=A.dtype)
        x = torch.empty((N, K), device=b.device, dtype=b.dtype)
        cholesky(A, L)
        solve(L, b, x)
        return x

    return fused_cholesky_solve
