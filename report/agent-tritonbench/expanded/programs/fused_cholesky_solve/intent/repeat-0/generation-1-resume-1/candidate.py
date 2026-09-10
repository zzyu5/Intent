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
    L: I.Out[I.f32, ("N", "N")],
    y: I.Out[I.f32, ("N", 1)],
    x: I.Out[I.f32, ("N", 1)],
):
    n = A.shape[0]
    rows = I.domain(0, n)
    columns = I.domain(0, n)
    L[rows, columns] = I.cast(0.0, I.f32)

    for i in I.domain(0, n):
        previous = I.domain(0, i)
        diagonal = A[i, i] - I.reduce.sum(
            L[i, previous] * L[i, previous],
            axis=0,
        )
        diagonal = _sqrt_positive(diagonal)
        L[i, i] = diagonal

        following = I.domain(i + 1, n)
        L[following, i] = I.fdiv(
            A[following, i]
            - I.reduce.sum(
                L[following, previous] * L[i, previous],
                axis=1,
            ),
            diagonal,
        )

    for i in I.domain(0, n):
        previous = I.domain(0, i)
        y[i, 0] = I.fdiv(
            b[i, 0]
            - I.reduce.sum(L[i, previous] * y[previous, 0], axis=0),
            L[i, i],
        )

    for offset in I.domain(0, n):
        i = n - 1 - offset
        following = I.domain(i + 1, n)
        x[i, 0] = I.fdiv(
            y[i, 0]
            - I.reduce.sum(L[following, i] * x[following, 0], axis=0),
            L[i, i],
        )


def build(context):
    compiled = context.compile("fused_cholesky_solve", _fused_cholesky_solve)

    def wrapper(A, b):
        L = torch.empty_like(A)
        y = torch.empty_like(b)
        x = torch.empty_like(b)
        compiled(A, b, L, y, x)
        return x

    return wrapper
