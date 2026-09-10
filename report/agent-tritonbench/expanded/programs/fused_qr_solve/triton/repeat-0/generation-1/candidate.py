import torch
import triton
import triton.language as tl


@triton.jit
def _copy_1d(src, dst, numel, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    values = tl.load(src + offsets, mask=mask, other=0.0)
    tl.store(dst + offsets, values, mask=mask)


@triton.jit
def _householder_step(
    a_ptr,
    b_ptr,
    k,
    M: tl.constexpr,
    N: tl.constexpr,
    NRHS: tl.constexpr,
    RHS_BLOCK: tl.constexpr,
):
    rows = tl.arange(0, M)
    active = rows >= k

    x = tl.load(a_ptr + rows * N + k, mask=active, other=0.0)
    x0 = tl.load(a_ptr + k * N + k)
    norm = tl.sqrt(tl.sum(x * x, axis=0))
    sign = tl.where(x0 >= 0.0, 1.0, -1.0)
    alpha = -sign * norm

    v = tl.where(rows == k, x0 - alpha, x)
    v_norm_sq = tl.sum(v * v, axis=0)
    beta = 2.0 / v_norm_sq

    tl.store(a_ptr + rows * N + k, tl.where(rows == k, alpha, v), mask=active)

    for j in tl.range(k + 1, N):
        column = tl.load(a_ptr + rows * N + j, mask=active, other=0.0)
        projection = tl.sum(v * column, axis=0)
        column = column - beta * v * projection
        tl.store(a_ptr + rows * N + j, column, mask=active)

    rhs_cols = tl.arange(0, RHS_BLOCK)
    rhs_mask = rhs_cols < NRHS
    rhs_offsets = rows[:, None] * NRHS + rhs_cols[None, :]
    rhs_valid = active[:, None] & rhs_mask[None, :]
    rhs = tl.load(b_ptr + rhs_offsets, mask=rhs_valid, other=0.0)
    projection = tl.sum(v[:, None] * rhs, axis=0)
    rhs = rhs - beta * v[:, None] * projection[None, :]
    tl.store(b_ptr + rhs_offsets, rhs, mask=rhs_valid)


@triton.jit
def _back_substitute(
    a_ptr,
    b_ptr,
    x_ptr,
    N: tl.constexpr,
    NRHS: tl.constexpr,
    RHS_BLOCK: tl.constexpr,
):
    rows = tl.arange(0, N)
    rhs_cols = tl.arange(0, RHS_BLOCK)
    rhs_mask = rhs_cols < NRHS

    for reverse_i in tl.range(0, N):
        i = N - 1 - reverse_i
        future = rows > i
        r = tl.load(a_ptr + i * N + rows, mask=future, other=0.0)
        future_x = tl.load(
            x_ptr + rows[:, None] * NRHS + rhs_cols[None, :],
            mask=future[:, None] & rhs_mask[None, :],
            other=0.0,
        )
        contribution = tl.sum(r[:, None] * future_x, axis=0)
        rhs = tl.load(b_ptr + i * NRHS + rhs_cols, mask=rhs_mask, other=0.0)
        diagonal = tl.load(a_ptr + i * N + i)
        value = (rhs - contribution) / diagonal
        tl.store(x_ptr + i * NRHS + rhs_cols, value, mask=rhs_mask)


def build(context):
    def fused_qr_solve(A, b):
        m = 512
        n = 256
        nrhs = 10

        a_work = torch.empty_like(A)
        b_work = torch.empty_like(b)

        _copy_1d[(triton.cdiv(A.numel(), 256),)](A, a_work, A.numel(), BLOCK=256)
        _copy_1d[(triton.cdiv(b.numel(), 256),)](b, b_work, b.numel(), BLOCK=256)

        for k in range(n):
            _householder_step[(1,)](
                a_work,
                b_work,
                k,
                M=m,
                N=n,
                NRHS=nrhs,
                RHS_BLOCK=16,
                num_warps=8,
            )

        x = torch.empty((n, nrhs), device=A.device, dtype=A.dtype)
        _back_substitute[(1,)](
            a_work,
            b_work,
            x,
            N=n,
            NRHS=nrhs,
            RHS_BLOCK=16,
            num_warps=8,
        )
        return x

    return fused_qr_solve
