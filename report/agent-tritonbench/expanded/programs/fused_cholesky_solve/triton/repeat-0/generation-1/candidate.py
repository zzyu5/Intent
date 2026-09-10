import torch
import triton
import triton.language as tl


@triton.jit
def _fused_cholesky_solve(
    a_ptr,
    b_ptr,
    l_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    cols = tl.arange(0, BLOCK)

    # Build one row of the Cholesky factor at a time.  All rows are handled by
    # one program so each row's data dependency is ordered without a grid
    # barrier.
    for i in range(N):
        row = tl.load(a_ptr + i * N + cols, mask=cols < N, other=0.0)
        for j in range(i):
            prior = tl.load(l_ptr + j * N + cols, mask=cols < j, other=0.0)
            dot = tl.sum(row * prior, axis=0)
            aij = tl.load(a_ptr + i * N + j)
            lj = tl.load(l_ptr + j * N + j)
            value = (aij - dot) / lj
            row = tl.where(cols == j, value, row)

        lower = tl.where(cols < i, row * row, 0.0)
        diagonal = tl.sqrt(tl.load(a_ptr + i * N + i) - tl.sum(lower, axis=0))
        row = tl.where(cols == i, diagonal, row)
        tl.store(l_ptr + i * N + cols, row, mask=cols <= i)

    # Forward substitution for L y = b.
    y = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(N):
        lower_row = tl.load(l_ptr + i * N + cols, mask=cols < i, other=0.0)
        dot = tl.sum(lower_row * y, axis=0)
        rhs = tl.load(b_ptr + i) - dot
        value = rhs / tl.load(l_ptr + i * N + i)
        y = tl.where(cols == i, value, y)

    # Backward substitution for L.T x = y.
    x = tl.zeros([BLOCK], dtype=tl.float32)
    for step in range(N):
        i = N - 1 - step
        column = tl.load(l_ptr + cols * N + i, mask=cols > i, other=0.0)
        dot = tl.sum(column * x, axis=0)
        rhs = tl.sum(tl.where(cols == i, y, 0.0), axis=0) - dot
        value = rhs / tl.load(l_ptr + i * N + i)
        x = tl.where(cols == i, value, x)

    tl.store(out_ptr + cols, x, mask=cols < N)


def build(context):
    def wrapper(A, b):
        l_factor = torch.empty_like(A)
        output = torch.empty_like(b)
        _fused_cholesky_solve[(1,)](
            A,
            b,
            l_factor,
            output,
            N=256,
            BLOCK=256,
            num_warps=8,
        )
        return output

    return wrapper
