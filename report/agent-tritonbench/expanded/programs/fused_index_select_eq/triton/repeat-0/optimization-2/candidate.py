import torch
import triton
import triton.language as tl


@triton.jit
def _fused_index_select_eq(
    input_ptr,
    index_ptr,
    other,
    output_ptr,
    BLOCK_ROWS: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = tl.arange(0, 128)

    selected_rows = tl.load(index_ptr + row_offsets).to(tl.int32)
    input_offsets = selected_rows[:, None] * 128 + col_offsets[None, :]
    values = tl.load(input_ptr + input_offsets)
    result = values == other
    output_offsets = row_offsets[:, None] * 128 + col_offsets[None, :]
    tl.store(output_ptr + output_offsets, result)


def build(context):
    def wrapper(input, dim, index, other, *, out=None):
        if out is None:
            out = torch.empty(
                (index.numel(), input.shape[1]),
                dtype=torch.bool,
                device=input.device,
            )

        n_rows = index.numel()
        _fused_index_select_eq[(triton.cdiv(n_rows, 16),)](
            input,
            index,
            other,
            out,
            BLOCK_ROWS=16,
            num_warps=8,
            num_stages=1,
        )
        return out

    return wrapper
