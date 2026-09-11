import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def log1p_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    result = libdevice.log1p(values)
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def wrapper(input, *, out=None):
        if out is None:
            out = torch.empty_like(input)
        n_elements = input.numel()
        if n_elements:
            log1p_kernel[(triton.cdiv(n_elements, 2048),)](
                input, out, n_elements, BLOCK=2048, num_warps=8
            )
        return out

    return wrapper
