import torch
import triton
import triton.language as tl


@triton.jit
def _fused_contiguous_kernel(
    input_ptr,
    index_ptr,
    output_ptr,
    other,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_COLS)

    selected_row = tl.load(index_ptr + row).to(tl.int32)
    input_offsets = selected_row * BLOCK_COLS + cols
    values = tl.load(input_ptr + input_offsets)
    result = (values == other).to(tl.int8)

    output_offsets = row * BLOCK_COLS + cols
    tl.store(output_ptr + output_offsets, result)


def launch(input, index, output, other):
    n_rows = index.shape[0]
    return _fused_contiguous_kernel[(n_rows,)](
        input,
        index,
        output,
        other,
        BLOCK_COLS=128,
        num_warps=1,
        num_stages=1,
    )


def run(input, index, other):
    output = torch.empty((index.shape[0], input.shape[1]), device=input.device, dtype=torch.int8)
    launch(input, index, output, other)
    return output
