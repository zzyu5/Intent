import torch
import triton
import triton.language as tl


@triton.jit
def _factor_diagonal(
    a_ptr,
    l_ptr,
    N: tl.constexpr,
    K0: tl.constexpr,
    BLOCK: tl.constexpr,
    FIRST: tl.constexpr,
):
    cols = tl.arange(0, BLOCK)

    # The diagonal panel is small enough that its row dependency can stay in
    # one program. Earlier panels have already reduced this block in l_ptr.
    for r in range(BLOCK):
        row_idx = K0 + r
        if FIRST:
            row = tl.load(a_ptr + row_idx * N + K0 + cols, mask=cols <= r, other=0.0)
        else:
            row = tl.load(l_ptr + row_idx * N + K0 + cols, mask=cols <= r, other=0.0)

        for q in range(r):
            prior = tl.load(l_ptr + (K0 + q) * N + K0 + cols, mask=cols < q, other=0.0)
            dot = tl.sum(row * prior, axis=0)
            if FIRST:
                residual = tl.load(a_ptr + row_idx * N + K0 + q)
            else:
                residual = tl.load(l_ptr + row_idx * N + K0 + q)
            pivot = tl.load(l_ptr + (K0 + q) * N + K0 + q)
            value = (residual - dot) / pivot
            row = tl.where(cols == q, value, row)

        if FIRST:
            diagonal = tl.load(a_ptr + row_idx * N + row_idx)
        else:
            diagonal = tl.load(l_ptr + row_idx * N + row_idx)
        lower_norm = tl.sum(tl.where(cols < r, row * row, 0.0), axis=0)
        value = tl.sqrt(diagonal - lower_norm)
        row = tl.where(cols == r, value, row)
        tl.store(l_ptr + row_idx * N + K0 + cols, row, mask=cols <= r)


@triton.jit
def _panel_trsm(
    a_ptr,
    l_ptr,
    N: tl.constexpr,
    K0: tl.constexpr,
    BLOCK: tl.constexpr,
    FIRST: tl.constexpr,
):
    row_idx = K0 + BLOCK + tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    if FIRST:
        values = tl.load(a_ptr + row_idx * N + K0 + cols)
    else:
        values = tl.load(l_ptr + row_idx * N + K0 + cols)

    # Solve X * L_panel.T = residual one row at a time across the panel.
    for q in range(BLOCK):
        prior = tl.load(l_ptr + (K0 + q) * N + K0 + cols, mask=cols < q, other=0.0)
        dot = tl.sum(values * prior, axis=0)
        residual = tl.sum(tl.where(cols == q, values, 0.0), axis=0)
        pivot = tl.load(l_ptr + (K0 + q) * N + K0 + q)
        value = (residual - dot) / pivot
        values = tl.where(cols == q, value, values)

    tl.store(l_ptr + row_idx * N + K0 + cols, values)


@triton.jit
def _update_trailing(
    a_ptr,
    l_ptr,
    N: tl.constexpr,
    K0: tl.constexpr,
    BLOCK: tl.constexpr,
    FIRST: tl.constexpr,
):
    block_i = tl.program_id(0)
    block_j = tl.program_id(1)
    row0 = K0 + BLOCK + block_i * BLOCK
    col0 = K0 + BLOCK + block_j * BLOCK

    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    depth = tl.arange(0, BLOCK)
    row_offsets = (row0 + rows)[:, None] * N + (K0 + depth)[None, :]
    col_offsets = (col0 + cols)[None, :] * N + (K0 + depth)[:, None]
    left = tl.load(l_ptr + row_offsets)
    right = tl.load(l_ptr + col_offsets)
    product = tl.dot(left, right, input_precision="ieee")

    valid = (row0 + rows)[:, None] < N
    valid = valid & ((col0 + cols)[None, :] < N)
    valid = valid & ((row0 + rows)[:, None] >= (col0 + cols)[None, :])
    offsets = (row0 + rows)[:, None] * N + (col0 + cols)[None, :]
    if FIRST:
        residual = tl.load(a_ptr + offsets, mask=valid, other=0.0)
    else:
        residual = tl.load(l_ptr + offsets, mask=valid, other=0.0)
    tl.store(l_ptr + offsets, residual - product, mask=valid)


@triton.jit
def _forward_block(
    l_ptr,
    b_ptr,
    out_ptr,
    N: tl.constexpr,
    K0: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    rows = tl.arange(0, BLOCK)
    rhs = tl.load(b_ptr + K0 + rows)

    for block_idx in range(K0 // BLOCK):
        col0 = block_idx * BLOCK
        matrix = tl.load(l_ptr + (K0 + rows)[:, None] * N + (col0 + rows)[None, :])
        previous = tl.load(out_ptr + col0 + rows)
        rhs -= tl.sum(matrix * previous[None, :], axis=1)

    values = tl.zeros([BLOCK], dtype=tl.float32)
    for r in range(BLOCK):
        row = tl.load(l_ptr + (K0 + r) * N + K0 + rows, mask=rows < r, other=0.0)
        dot = tl.sum(row * values, axis=0)
        residual = tl.sum(tl.where(rows == r, rhs, 0.0), axis=0)
        pivot = tl.load(l_ptr + (K0 + r) * N + K0 + r)
        value = (residual - dot) / pivot
        values = tl.where(rows == r, value, values)

    tl.store(out_ptr + K0 + rows, values)


@triton.jit
def _backward_block(
    l_ptr,
    out_ptr,
    N: tl.constexpr,
    K0: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    cols = tl.arange(0, BLOCK)
    rhs = tl.load(out_ptr + K0 + cols)

    for block_idx in range(K0 // BLOCK + 1, NUM_BLOCKS):
        row0 = block_idx * BLOCK
        matrix = tl.load(l_ptr + (row0 + cols)[:, None] * N + (K0 + cols)[None, :])
        previous = tl.load(out_ptr + row0 + cols)
        rhs -= tl.sum(matrix * previous[:, None], axis=0)

    values = tl.zeros([BLOCK], dtype=tl.float32)
    for step in range(BLOCK):
        r = BLOCK - 1 - step
        column = tl.load(l_ptr + (K0 + cols) * N + K0 + r, mask=cols > r, other=0.0)
        dot = tl.sum(column * values, axis=0)
        residual = tl.sum(tl.where(cols == r, rhs, 0.0), axis=0)
        pivot = tl.load(l_ptr + (K0 + r) * N + K0 + r)
        value = (residual - dot) / pivot
        values = tl.where(cols == r, value, values)

    tl.store(out_ptr + K0 + cols, values)


def build(context):
    del context

    def wrapper(A, b):
        l_factor = torch.empty_like(A)
        output = torch.empty_like(b)

        n = 256
        block = 64
        num_blocks = n // block

        for block_idx in range(num_blocks):
            k0 = block_idx * block
            first = block_idx == 0
            _factor_diagonal[(1,)](
                A,
                l_factor,
                N=n,
                K0=k0,
                BLOCK=block,
                FIRST=first,
                num_warps=4,
            )
            if block_idx < num_blocks - 1:
                below_rows = n - k0 - block
                tail_blocks = num_blocks - block_idx - 1
                _panel_trsm[(below_rows,)](
                    A,
                    l_factor,
                    N=n,
                    K0=k0,
                    BLOCK=block,
                    FIRST=first,
                    num_warps=4,
                )
                _update_trailing[(tail_blocks, tail_blocks)](
                    A,
                    l_factor,
                    N=n,
                    K0=k0,
                    BLOCK=block,
                    FIRST=first,
                    num_warps=8,
                )

        for block_idx in range(num_blocks):
            _forward_block[(1,)](
                l_factor,
                b,
                output,
                N=n,
                K0=block_idx * block,
                BLOCK=block,
                NUM_BLOCKS=num_blocks,
                num_warps=4,
            )

        for block_idx in range(num_blocks - 1, -1, -1):
            _backward_block[(1,)](
                l_factor,
                output,
                N=n,
                K0=block_idx * block,
                BLOCK=block,
                NUM_BLOCKS=num_blocks,
                num_warps=4,
            )

        return output

    return wrapper
