import torch
import intent
import intent.language as I


@intent.fn
def _sqrt_positive(value):
    estimate = I.maximum(value, I.cast(1.0, I.f32))
    for _ in I.domain(0, 32):
        estimate = I.fdiv(
            estimate + I.fdiv(value, estimate),
            I.cast(2.0, I.f32),
        )
    return estimate


@intent.kernel
def _fused_cholesky_solve(
    A: I.In[I.f32, ("N", "N")],
    b: I.In[I.f32, ("N", 1)],
    L: I.InOut[I.f32, ("N", "N")],
    y: I.InOut[I.f32, ("N", 1)],
    x: I.InOut[I.f32, ("N", 1)],
):
    n = A.shape[0]

    for i in I.domain(0, n):
        diagonal = A[i, i]
        for j in I.domain(0, i):
            diagonal = diagonal - L[i, j] * L[i, j]
        diagonal = _sqrt_positive(diagonal)
        L[i, i] = diagonal

        for row in I.domain(i + 1, n):
            value = A[row, i]
            for j in I.domain(0, i):
                value = value - L[row, j] * L[i, j]
            L[row, i] = I.fdiv(value, diagonal)

    for i in I.domain(0, n):
        value = b[i, 0]
        for j in I.domain(0, i):
            value = value - L[i, j] * y[j, 0]
        y[i, 0] = I.fdiv(value, L[i, i])

    for offset in I.domain(0, n):
        i = n - 1 - offset
        value = y[i, 0]
        for j in I.domain(i + 1, n):
            value = value - L[j, i] * x[j, 0]
        x[i, 0] = I.fdiv(value, L[i, i])


def build(context):
    compiled = context.compile("fused_cholesky_solve", _fused_cholesky_solve)

    def wrapper(A, b):
        L = torch.empty_like(A)
        y = torch.empty_like(b)
        x = torch.empty_like(b)
        compiled(A, b, L, y, x)
        return x

    return wrapper
