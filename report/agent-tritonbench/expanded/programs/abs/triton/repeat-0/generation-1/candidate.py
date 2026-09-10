import torch
import triton
import triton.language as tl


@triton.jit
def abs_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, tl.abs(values), mask=mask)


def build(context):
    def wrapper(input, *, out=None):
        if out is None:
            out = torch.empty_like(input)
        count = input.numel()
        abs_kernel[(triton.cdiv(count, 1024),)](
            input,
            out,
            count,
            BLOCK=1024,
            num_warps=4,
        )
        return out

    return wrapper
