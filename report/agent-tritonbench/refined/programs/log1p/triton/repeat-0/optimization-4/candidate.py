import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def log1p_kernel(input_ptr, output_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input_ptr + offsets)
    tl.store(output_ptr + offsets, libdevice.log1p(values))


def build(context):
    def wrapper(input, *, out=None):
        if out is None:
            out = torch.empty_like(input)
        log1p_kernel[(2048,)](
            input, out, BLOCK=512, num_warps=2, num_stages=3
        )
        return out

    return wrapper
