import torch
import triton
import triton.language as tl


@triton.jit
def floor_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, tl.floor(values), mask=mask)


def build(context):
    def wrapper(input, *, out=None):
        if out is None:
            out = torch.empty_like(input)

        n_elements = input.numel()
        floor_kernel[(triton.cdiv(n_elements, 1024),)](
            input,
            out,
            n_elements,
            BLOCK_SIZE=1024,
            num_warps=4,
        )
        return out

    return wrapper
