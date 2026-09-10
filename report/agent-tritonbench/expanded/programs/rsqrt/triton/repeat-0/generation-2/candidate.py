import torch
import triton
import triton.language as tl


@triton.jit
def rsqrt_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, tl.rsqrt(values), mask=mask)


def build(context):
    def wrapper(input, *, out=None):
        output = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        rsqrt_kernel[(triton.cdiv(n_elements, 256),)](input, output, n_elements, BLOCK=256)
        return output

    return wrapper
