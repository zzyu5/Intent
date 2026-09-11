import torch
import triton
import triton.language as tl


@triton.jit
def _copy_lower(src, dst, n, tiles, ss0, ss1, ds0, ds1, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid // tiles * BLOCK + tl.arange(0, BLOCK)
    cols = pid % tiles * BLOCK + tl.arange(0, BLOCK)
    mask = (rows[:, None] < n) & (cols[None, :] < n) & (rows[:, None] >= cols[None, :])
    value = tl.load(src + rows[:, None] * ss0 + cols[None, :] * ss1, mask=mask, other=0.0)
    tl.store(dst + rows[:, None] * ds0 + cols[None, :] * ds1, value, mask=mask)


@triton.jit
def _chol_diagonal(
    factor,
    start,
    sf0,
    sf1,
    BLOCK: tl.constexpr,
):
    rows = tl.arange(0, BLOCK)

    for i in range(BLOCK):
        active = rows >= i
        residual = tl.load(
            factor + (start + rows) * sf0 + (start + i) * sf1,
            mask=active,
            other=0.0,
        )
        diagonal_input = tl.load(
            factor + (start + i) * sf0 + (start + i) * sf1
        )

        accumulated = tl.zeros((BLOCK,), dtype=tl.float32)
        diagonal_accumulated = 0.0
        for k in range(i):
            left = tl.load(
                factor + (start + rows) * sf0 + (start + k) * sf1,
                mask=active,
                other=0.0,
            )
            right = tl.load(
                factor + (start + i) * sf0 + (start + k) * sf1
            )
            accumulated += left * right
            diagonal_accumulated += right * right

        diagonal = tl.sqrt(diagonal_input - diagonal_accumulated)
        value = tl.where(rows == i, diagonal, (residual - accumulated) / diagonal)
        tl.store(
            factor + (start + rows) * sf0 + (start + i) * sf1,
            value,
            mask=active,
        )


@triton.jit
def _chol_panel(
    factor,
    start,
    n,
    sf0,
    sf1,
    BLOCK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = start + BLOCK + pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < n

    for j in range(BLOCK):
        accumulated = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
        for k in range(j):
            left = tl.load(
                factor + rows * sf0 + (start + k) * sf1,
                mask=row_mask,
                other=0.0,
            )
            right = tl.load(
                factor + (start + j) * sf0 + (start + k) * sf1
            )
            accumulated += left * right

        pointer = factor + rows * sf0 + (start + j) * sf1
        numerator = tl.load(pointer, mask=row_mask, other=0.0) - accumulated
        diagonal = tl.load(
            factor + (start + j) * sf0 + (start + j) * sf1
        )
        tl.store(pointer, numerator / diagonal, mask=row_mask)


@triton.jit
def _chol_update(
    factor,
    start,
    n,
    tiles,
    sf0,
    sf1,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    tile_row = pid // tiles
    tile_col = pid % tiles
    end = start + BLOCK
    rows = end + tile_row * BLOCK + tl.arange(0, BLOCK)
    cols = end + tile_col * BLOCK + tl.arange(0, BLOCK)
    row_mask = rows < n
    col_mask = cols < n
    k = start + tl.arange(0, BLOCK)

    left = tl.load(
        factor + rows[:, None] * sf0 + k[None, :] * sf1,
        mask=row_mask[:, None],
        other=0.0,
    )
    right = tl.load(
        factor + cols[:, None] * sf0 + k[None, :] * sf1,
        mask=col_mask[:, None],
        other=0.0,
    )
    update = tl.dot(left, tl.trans(right), input_precision="ieee")

    pointer = factor + rows[:, None] * sf0 + cols[None, :] * sf1
    store_mask = row_mask[:, None] & col_mask[None, :] & (rows[:, None] >= cols[None, :])
    old = tl.load(pointer, mask=store_mask, other=0.0)
    tl.store(pointer, old - update, mask=store_mask)


@triton.jit
def _forward_update(
    factor,
    rhs,
    output,
    start,
    n,
    sf0,
    sf1,
    sr0,
    so0,
    USE_RHS: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = start + BLOCK + pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < n
    cols = start + tl.arange(0, BLOCK)
    left = tl.load(
        factor + rows[:, None] * sf0 + cols[None, :] * sf1,
        mask=row_mask[:, None],
        other=0.0,
    )
    solved = tl.load(output + cols * so0)
    update = tl.sum(left * solved[None, :], axis=1)
    if USE_RHS:
        old = tl.load(rhs + rows * sr0, mask=row_mask, other=0.0)
    else:
        old = tl.load(output + rows * so0, mask=row_mask, other=0.0)
    tl.store(output + rows * so0, old - update, mask=row_mask)


@triton.jit
def _forward_diagonal(
    factor,
    rhs,
    output,
    start,
    sf0,
    sf1,
    sr0,
    so0,
    USE_RHS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for i in range(BLOCK):
        accumulated = 0.0
        for k in range(i):
            left = tl.load(
                factor + (start + i) * sf0 + (start + k) * sf1
            )
            solved = tl.load(output + (start + k) * so0)
            accumulated += left * solved
        if USE_RHS:
            value = tl.load(rhs + (start + i) * sr0) - accumulated
        else:
            value = tl.load(output + (start + i) * so0) - accumulated
        diagonal = tl.load(
            factor + (start + i) * sf0 + (start + i) * sf1
        )
        tl.store(output + (start + i) * so0, value / diagonal)


@triton.jit
def _backward_update(
    factor,
    output,
    start,
    sf0,
    sf1,
    so0,
    BLOCK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < start
    cols = start + tl.arange(0, BLOCK)
    left = tl.load(
        factor + cols[None, :] * sf0 + rows[:, None] * sf1,
        mask=row_mask[:, None],
        other=0.0,
    )
    solved = tl.load(output + cols * so0)
    update = tl.sum(left * solved[None, :], axis=1)
    pointer = output + rows * so0
    value = tl.load(pointer, mask=row_mask, other=0.0)
    tl.store(pointer, value - update, mask=row_mask)


@triton.jit
def _backward_diagonal(
    factor,
    output,
    start,
    sf0,
    sf1,
    so0,
    BLOCK: tl.constexpr,
):
    for i in range(BLOCK - 1, -1, -1):
        accumulated = 0.0
        for j in range(i + 1, BLOCK):
            left = tl.load(
                factor + (start + j) * sf0 + (start + i) * sf1
            )
            solved = tl.load(output + (start + j) * so0)
            accumulated += left * solved
        pointer = output + (start + i) * so0
        value = tl.load(pointer) - accumulated
        diagonal = tl.load(
            factor + (start + i) * sf0 + (start + i) * sf1
        )
        tl.store(pointer, value / diagonal)


def build(context):
    del context

    def wrapper(A, b):
        n = A.shape[0]
        block = 32
        factor = torch.empty_like(A)
        output = torch.empty_like(b)
        tile_count = triton.cdiv(n, block)

        _copy_lower[(tile_count * tile_count,)](
            A,
            factor,
            n,
            tile_count,
            A.stride(0),
            A.stride(1),
            factor.stride(0),
            factor.stride(1),
            BLOCK=block,
            num_warps=4,
        )

        for start in range(0, n, block):
            _chol_diagonal[(1,)](
                factor,
                start,
                factor.stride(0),
                factor.stride(1),
                BLOCK=block,
                num_warps=4,
            )
            end = start + block
            if end < n:
                tail = n - end
                tail_tiles = triton.cdiv(tail, block)
                _chol_panel[(triton.cdiv(tail, 8),)](
                    factor,
                    start,
                    n,
                    factor.stride(0),
                    factor.stride(1),
                    BLOCK=block,
                    BLOCK_ROWS=8,
                    num_warps=2,
                )
                _chol_update[(tail_tiles * tail_tiles,)](
                    factor,
                    start,
                    n,
                    tail_tiles,
                    factor.stride(0),
                    factor.stride(1),
                    BLOCK=block,
                    num_warps=4,
                )

        for start in range(0, n, block):
            _forward_diagonal[(1,)](
                factor,
                b,
                output,
                start,
                factor.stride(0),
                factor.stride(1),
                b.stride(0),
                output.stride(0),
                USE_RHS=(start == 0),
                BLOCK=block,
                num_warps=1,
            )
            end = start + block
            if end < n:
                _forward_update[(triton.cdiv(n - end, 32),)](
                    factor,
                    b,
                    output,
                    start,
                    n,
                    factor.stride(0),
                    factor.stride(1),
                    b.stride(0),
                    output.stride(0),
                    USE_RHS=(start == 0),
                    BLOCK=block,
                    BLOCK_ROWS=32,
                    num_warps=4,
                )

        for start in range(n - block, -1, -block):
            _backward_diagonal[(1,)](
                factor,
                output,
                start,
                factor.stride(0),
                factor.stride(1),
                output.stride(0),
                BLOCK=block,
                num_warps=1,
            )
            if start > 0:
                _backward_update[(triton.cdiv(start, 32),)](
                    factor,
                    output,
                    start,
                    factor.stride(0),
                    factor.stride(1),
                    output.stride(0),
                    BLOCK=block,
                    BLOCK_ROWS=32,
                    num_warps=4,
                )

        return output

    return wrapper
