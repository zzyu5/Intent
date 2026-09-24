import torch
import intent
import intent.language as I


@intent.kernel
def _qr_factor(
    A: I.In[I.f32, (64, 16)],
    Q: I.Out[I.f32, (64, 16)],
    R: I.Out[I.f32, (16, 16)],
):
    rows = I.domain(0, 64)
    columns = I.domain(0, 16)

    # The residual columns in Q are updated in place as the orthogonal basis
    # is formed.  The outer loop is ordered because each column depends on
    # the previously normalized basis columns.
    Q[rows, columns] = A[rows, columns]
    R[columns, columns] = I.zeros((16, 16), I.f32)
    for j in I.domain(0, 16):
        qj = Q[rows, j]
        norm_sq = I.reduce.sum(qj * qj, axis=0, acc_dtype=I.f32)
        norm = I.sqrt(norm_sq)
        qj = I.fdiv(qj, norm)
        Q[rows, j] = qj
        for col in I.domain(j + 1, 16):
            v = Q[rows, col]
            rjc = I.reduce.sum(qj * v, axis=0, acc_dtype=I.f32)
            R[j, col] = rjc
            Q[rows, col] = v - qj * rjc
        R[j, j] = norm


@intent.kernel
def _qr_solve(
    Q: I.In[I.f32, (64, 16)],
    R: I.In[I.f32, (16, 16)],
    b: I.In[I.f32, (64, 1)],
    out: I.Out[I.f32, (16, 1)],
):
    rows = I.domain(0, 64)
    columns = I.domain(0, 16)
    column_indices = I.indices(columns)
    y = I.buffer((16,), I.f32, init=I.zeros((16,), I.f32))
    x = I.buffer((16,), I.f32, init=I.zeros((16,), I.f32))

    # Each right-hand-side projection is independent once Q is complete.
    for j in I.parallel(columns):
        y[j] = I.reduce.sum(Q[rows, j] * b[rows, 0], axis=0, acc_dtype=I.f32)

    # Back substitution is the second ordered dependency chain.
    for ordinal in I.domain(0, 16):
        j = 15 - ordinal
        tail = I.reduce.sum(
            I.select(
                column_indices > j,
                R[j, column_indices] * x[column_indices],
                I.cast(0.0, I.f32),
            ),
            axis=0,
            acc_dtype=I.f32,
        )
        xj = I.fdiv(y[j] - tail, R[j, j])
        x[j] = xj

    for j in I.parallel(columns):
        out[j, 0] = x[j]


def build(context):
    factor = context.compile("least_squares_qr_factor", _qr_factor)
    solve = context.compile("least_squares_qr_solve", _qr_solve)

    def least_squares_qr(
        A: torch.Tensor,
        b: torch.Tensor,
        *,
        mode: str = "reduced",
        out: torch.Tensor = None,
    ) -> torch.Tensor:
        Q = torch.empty((64, 16), device=A.device, dtype=A.dtype)
        R = torch.empty((16, 16), device=A.device, dtype=A.dtype)
        result = out if out is not None else torch.empty(
            (16, 1), device=A.device, dtype=A.dtype
        )
        factor(A, Q, R)
        solve(Q, R, b, result)
        return result

    return least_squares_qr
