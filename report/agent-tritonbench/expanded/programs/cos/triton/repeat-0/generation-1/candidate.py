import torch
import triton
import triton.language as tl


@triton.jit
def cos_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(axis=0)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, tl.cos(values), mask=mask)


def build(context):
    def wrapper(input, *, out=None):
        if out is None:
            out = torch.empty_like(input)

        n_elements = input.numel()
        cos_kernel[(triton.cdiv(n_elements, 1024),)](
            input,
            out,
            n_elements,
            BLOCK_SIZE=1024,
            num_warps=4,
        )
        return out

    return wrapper
