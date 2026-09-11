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

    # Keep the dependency chain in one program. Each iteration computes one
    # row of the lower factor from rows that have already been written.
    for i in range(BLOCK):
        row = tl.load(a_ptr + i * N + cols, mask=cols <= i, other=0.0)
        for j in range(i):
            prior = tl.load(l_ptr + j * N + cols, mask=cols < j, other=0.0)
            dot = tl.sum(row * prior, axis=0)
            residual = tl.sum(tl.where(cols == j, row, 0.0), axis=0)
            pivot = tl.load(l_ptr + j * N + j)
            value = (residual - dot) / pivot
            row = tl.where(cols == j, value, row)

        diagonal = tl.load(a_ptr + i * N + i)
        lower_norm = tl.sum(tl.where(cols < i, row * row, 0.0), axis=0)
        diagonal = tl.sqrt(diagonal - lower_norm)
        row = tl.where(cols == i, diagonal, row)
        tl.store(l_ptr + i * N + cols, row, mask=cols <= i)

    # Forward substitution: the vector lanes retain the solved entries.
    forward = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(BLOCK):
        row = tl.load(l_ptr + i * N + cols, mask=cols < i, other=0.0)
        dot = tl.sum(row * forward, axis=0)
        rhs = tl.load(b_ptr + i)
        pivot = tl.load(l_ptr + i * N + i)
        value = (rhs - dot) / pivot
        forward = tl.where(cols == i, value, forward)

    # Backward substitution against L.T.
    solution = tl.zeros([BLOCK], dtype=tl.float32)
    for step in range(BLOCK):
        i = BLOCK - 1 - step
        column = tl.load(l_ptr + cols * N + i, mask=cols > i, other=0.0)
        dot = tl.sum(column * solution, axis=0)
        rhs = tl.sum(tl.where(cols == i, forward, 0.0), axis=0)
        pivot = tl.load(l_ptr + i * N + i)
        value = (rhs - dot) / pivot
        solution = tl.where(cols == i, value, solution)

    tl.store(out_ptr + cols, solution)


def build(context):
    del context

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
