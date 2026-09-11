import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def log1p_kernel(input_ptr, output_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input_ptr + offsets)
    result = libdevice.log1p(values)
    tl.store(output_ptr + offsets, result)


def build(context):
    def wrapper(input, *, out=None):
        if out is None:
            out = torch.empty_like(input)
        log1p_kernel[(512,)](input, out, BLOCK=2048, num_warps=8)
        return out

    return wrapper
