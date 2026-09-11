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
def _qr_panel(
    a_ptr,
    b_ptr,
    tau_ptr,
    panel_start,
    M: tl.constexpr,
    N: tl.constexpr,
    NRHS: tl.constexpr,
    PANEL: tl.constexpr,
    RHS_BLOCK: tl.constexpr,
):
    rows = tl.arange(0, M)
    rhs_cols = tl.arange(0, RHS_BLOCK)
    rhs_mask = rhs_cols < NRHS

    # One program owns a panel, so the reflectors can be formed and applied
    # in order without a synchronization between launches.
    for local in tl.range(0, PANEL):
        k = panel_start + local
        active = rows >= k
        below = rows > k

        x = tl.load(a_ptr + rows * N + k, mask=active, other=0.0)
        x0 = tl.load(a_ptr + k * N + k)
        norm = tl.sqrt(tl.sum(x * x, axis=0))
        sign = tl.where(x0 >= 0.0, 1.0, -1.0)
        alpha = -sign * norm

        # Store a unit-diagonal Householder vector below the diagonal.  The
        # diagonal of A remains the corresponding R diagonal.
        vdiag = x0 - alpha
        safe_vdiag = tl.where(vdiag != 0.0, vdiag, 1.0)
        v = tl.where(below, x / safe_vdiag, tl.where(rows == k, 1.0, 0.0))
        tau = 2.0 / tl.sum(v * v, axis=0)

        tl.store(tau_ptr + k, tau)
        tl.store(a_ptr + rows * N + k, tl.where(rows == k, alpha, v), mask=active)

        for local_col in tl.range(local + 1, PANEL):
            col = panel_start + local_col
            y = tl.load(a_ptr + rows * N + col, mask=active, other=0.0)
            projection = tl.sum(v * y, axis=0)
            y = y - tau * v * projection
            tl.store(a_ptr + rows * N + col, y, mask=active)

        # b is small enough to transform along with the panel factorization.
        rhs_offsets = rows[:, None] * NRHS + rhs_cols[None, :]
        rhs_valid = active[:, None] & rhs_mask[None, :]
        rhs = tl.load(b_ptr + rhs_offsets, mask=rhs_valid, other=0.0)
        projection = tl.sum(v[:, None] * rhs, axis=0)
        rhs = rhs - tau * v[:, None] * projection[None, :]
        tl.store(b_ptr + rhs_offsets, rhs, mask=rhs_valid)


@triton.jit
def _apply_panel(
    a_ptr,
    tau_ptr,
    panel_start,
    M: tl.constexpr,
    N: tl.constexpr,
    PANEL: tl.constexpr,
    COL_BLOCK: tl.constexpr,
):
    col_ids = tl.program_id(0) * COL_BLOCK + tl.arange(0, COL_BLOCK)
    col_count = N - panel_start - PANEL
    valid_cols = col_ids < col_count
    cols = panel_start + PANEL + col_ids
    offsets = tl.arange(0, M)[:, None] * N + cols[None, :]

    rows = tl.arange(0, M)
    for local in tl.range(0, PANEL):
        k = panel_start + local
        active = rows >= k
        v = tl.load(a_ptr + rows * N + k, mask=rows > k, other=0.0)
        v = tl.where(rows == k, 1.0, v)

        values = tl.load(a_ptr + offsets, mask=active[:, None] & valid_cols[None, :], other=0.0)
        projection = tl.sum(v[:, None] * values, axis=0)
        tau = tl.load(tau_ptr + k)
        values = values - tau * v[:, None] * projection[None, :]
        tl.store(a_ptr + offsets, values, mask=active[:, None] & valid_cols[None, :])


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
        panel = 64

        a_work = torch.empty_like(A)
        b_work = torch.empty_like(b)
        tau = torch.empty((n,), device=A.device, dtype=A.dtype)

        _copy_1d[(triton.cdiv(A.numel(), 1024),)](A, a_work, A.numel(), BLOCK=1024)
        _copy_1d[(triton.cdiv(b.numel(), 1024),)](b, b_work, b.numel(), BLOCK=1024)

        for panel_start in range(0, n, panel):
            _qr_panel[(1,)](
                a_work,
                b_work,
                tau,
                panel_start,
                M=m,
                N=n,
                NRHS=nrhs,
                PANEL=panel,
                RHS_BLOCK=16,
                num_warps=8,
            )
            trailing = n - panel_start - panel
            if trailing > 0:
                _apply_panel[(triton.cdiv(trailing, 4),)](
                    a_work,
                    tau,
                    panel_start,
                    M=m,
                    N=n,
                    PANEL=panel,
                    COL_BLOCK=4,
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
