import torch
import triton
import triton.language as tl


@triton.jit
def _gather_masked_fill(
    input_ptr,
    index_ptr,
    mask_ptr,
    output_ptr,
    value,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    offsets = pid * BLOCK + cols

    selected = tl.load(mask_ptr + offsets)
    active = ~selected
    indices = tl.load(index_ptr + offsets, mask=active, other=0).to(tl.int32)
    input_offsets = indices * N_COLS + cols
    gathered = tl.load(input_ptr + input_offsets, mask=active, other=0.0)
    result = tl.where(selected, value, gathered)
    tl.store(output_ptr + offsets, result)


def launch(input, index, mask, output, value):
    _gather_masked_fill[(1024,)](
        input,
        index,
        mask,
        output,
        value,
        N_COLS=1024,
        BLOCK=1024,
        num_warps=4,
        num_stages=2,
    )


def run(input, index, mask, value):
    output = torch.empty_like(input)
    launch(input, index, mask, output, value)
    return output
