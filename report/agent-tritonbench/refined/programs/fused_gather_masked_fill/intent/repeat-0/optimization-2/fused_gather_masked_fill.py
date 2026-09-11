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
    n_elements,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    offsets = pid * BLOCK + cols
    valid = offsets < n_elements

    selected = tl.load(mask_ptr + offsets, mask=valid, other=0)
    active = valid & ~selected
    indices = tl.load(index_ptr + offsets, mask=active, other=0)
    input_offsets = indices * N_COLS + cols
    gathered = tl.load(input_ptr + input_offsets, mask=active, other=0.0)
    result = tl.where(selected, value, gathered)
    tl.store(output_ptr + offsets, result, mask=valid)


def launch(input, index, mask, output, value):
    n_elements = input.numel()
    block = 1024
    _gather_masked_fill[(triton.cdiv(n_elements, block),)](
        input,
        index,
        mask,
        output,
        value,
        n_elements,
        N_COLS=1024,
        BLOCK=block,
        num_warps=4,
        num_stages=2,
    )


def run(input, index, mask, value):
    output = torch.empty_like(input)
    launch(input, index, mask, output, value)
    return output
