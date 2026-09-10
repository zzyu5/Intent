import torch
import triton
import triton.language as tl


@triton.jit
def abs_kernel(x, output, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask)
    tl.store(output + offsets, tl.abs(values), mask=mask)


def build(context):
    def wrapper(x):
        output = torch.empty_like(x)
        n_elements = x.numel()
        abs_kernel[(triton.cdiv(n_elements, 256),)](
            x, output, n_elements, BLOCK=256
        )
        return output

    return wrapper
