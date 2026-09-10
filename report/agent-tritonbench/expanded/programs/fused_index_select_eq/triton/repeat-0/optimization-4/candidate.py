import torch
import triton
import triton.language as tl


@triton.jit
def _fused_index_select_eq(input_ptr, index_ptr, other, output_ptr):
    row = tl.program_id(0)
    col_offsets = tl.arange(0, 128)

    selected_row = tl.load(index_ptr + row).to(tl.int32)
    values = tl.load(input_ptr + selected_row * 128 + col_offsets)
    tl.store(output_ptr + row * 128 + col_offsets, values == other)


def build(context):
    def wrapper(input, dim, index, other, *, out=None):
        if out is None:
            out = torch.empty(
                (index.numel(), input.shape[1]),
                dtype=torch.bool,
                device=input.device,
            )

        n_rows = index.numel()
        _fused_index_select_eq[(n_rows,)](
            input,
            index,
            other,
            out,
            num_warps=4,
            num_stages=1,
        )
        return out

    return wrapper
