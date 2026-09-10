import torch
import triton
import triton.language as tl


@triton.jit
def vector_add(x, y, output, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    lhs = tl.load(x + offsets, mask=valid)
    rhs = tl.load(y + offsets, mask=valid)
    tl.store(output + offsets, lhs + rhs, mask=valid)


def build(context):
    def wrapper(x, y):
        output = torch.empty_like(x)
        count = x.numel()
        vector_add[(triton.cdiv(count, 256),)](x, y, output, count, BLOCK=256)
        return output

    return wrapper
