import torch
import triton
import triton.language as tl


@triton.jit
def _fused_cholesky_solve(
    matrix,
    rhs,
    factor,
    output,
    n,
    sm0,
    sm1,
    sr0,
    sf0,
    sf1,
    so0,
    BLOCK: tl.constexpr,
):
    rows = tl.arange(0, BLOCK)
    valid_rows = rows < n

    # One CTA owns the dependency chain. Each lane carries one matrix row
    # through the column-wise Cholesky factorization.
    for i in range(BLOCK):
        active = valid_rows & (rows >= i)
        residual = tl.load(
            matrix + rows * sm0 + i * sm1,
            mask=active,
            other=0.0,
        )
        accumulated = tl.zeros((BLOCK,), dtype=tl.float32)
        diagonal_accumulated = 0.0
        for k in range(i):
            left = tl.load(
                factor + rows * sf0 + k * sf1,
                mask=active,
                other=0.0,
            )
            right = tl.load(factor + i * sf0 + k * sf1)
            accumulated += left * right
            diagonal_accumulated += right * right

        diagonal = tl.sqrt(
            tl.load(matrix + i * sm0 + i * sm1) - diagonal_accumulated
        )
        value = tl.where(rows == i, diagonal, (residual - accumulated) / diagonal)
        tl.store(
            factor + rows * sf0 + i * sf1,
            value,
            mask=active,
        )
        tl.debug_barrier()

    # Keep the forward residuals in the lanes instead of materializing an
    # intermediate vector. A reduction extracts the one active row each step.
    residual = tl.load(rhs + rows * sr0, mask=valid_rows, other=0.0)
    for i in range(BLOCK):
        active = valid_rows & (rows >= i)
        current = tl.sum(tl.where(rows == i, residual, 0.0), axis=0)
        diagonal = tl.load(factor + i * sf0 + i * sf1)
        solved = tl.fdiv(current, diagonal, ieee_rounding=True)
        residual = tl.where(rows == i, solved, residual)
        coefficient = tl.load(
            factor + rows * sf0 + i * sf1,
            mask=active,
            other=0.0,
        )
        residual = tl.where(rows > i, residual - coefficient * solved, residual)

    # Reverse substitution for L.T uses the same register-resident vector.
    for i in range(BLOCK - 1, -1, -1):
        current = tl.sum(tl.where(rows == i, residual, 0.0), axis=0)
        diagonal = tl.load(factor + i * sf0 + i * sf1)
        solved = tl.fdiv(current, diagonal, ieee_rounding=True)
        residual = tl.where(rows == i, solved, residual)
        coefficient = tl.load(
            factor + i * sf0 + rows * sf1,
            mask=valid_rows & (rows < i),
            other=0.0,
        )
        residual = tl.where(rows < i, residual - coefficient * solved, residual)

    tl.store(output + rows * so0, residual, mask=valid_rows)


def build(context):
    del context

    def wrapper(A, b):
        n = A.shape[0]
        factor = torch.empty_like(A)
        output = torch.empty_like(b)
        _fused_cholesky_solve[(1,)](
            A,
            b,
            factor,
            output,
            n,
            A.stride(0),
            A.stride(1),
            b.stride(0),
            factor.stride(0),
            factor.stride(1),
            output.stride(0),
            BLOCK=256,
            num_warps=8,
            num_stages=1,
        )
        return output

    return wrapper
