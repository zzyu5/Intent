import torch
import triton
import triton.language as tl


@triton.jit
def _abs_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, tl.abs(x), mask=mask)


def build(context):
    def wrapper(x):
        out = torch.empty_like(x)
        n_elements = x.numel()
        if n_elements:
            grid = (triton.cdiv(n_elements, 1024),)
            _abs_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
        return out

    return wrapper
