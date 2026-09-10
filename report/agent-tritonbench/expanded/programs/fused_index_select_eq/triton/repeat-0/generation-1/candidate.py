import torch
import triton
import triton.language as tl


@triton.jit
def _fused_index_select_eq(
    input_ptr,
    index_ptr,
    other,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = tl.arange(0, BLOCK_COLS)
    row_mask = row_offsets < n_rows
    col_mask = col_offsets < n_cols

    selected_rows = tl.load(index_ptr + row_offsets, mask=row_mask, other=0)
    input_offsets = selected_rows[:, None] * n_cols + col_offsets[None, :]
    values = tl.load(
        input_ptr + input_offsets,
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    result = values == other
    output_offsets = row_offsets[:, None] * n_cols + col_offsets[None, :]
    tl.store(
        output_ptr + output_offsets,
        result,
        mask=row_mask[:, None] & col_mask[None, :],
    )


def build(context):
    def wrapper(input, dim, index, other, *, out=None):
        if out is None:
            out = torch.empty(
                (index.numel(), input.shape[1]),
                dtype=torch.bool,
                device=input.device,
            )

        n_rows = index.numel()
        n_cols = input.shape[1]
        _fused_index_select_eq[(triton.cdiv(n_rows, 4),)](
            input,
            index,
            other,
            out,
            n_rows,
            n_cols,
            BLOCK_ROWS=4,
            BLOCK_COLS=128,
            num_warps=4,
        )
        return out

    return wrapper
