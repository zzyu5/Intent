import torch
import triton
import triton.language as tl


@triton.jit
def _abs_full_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    tl.store(out_ptr + offsets, tl.abs(x))


@triton.jit
def _abs_tail_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, tl.abs(x), mask=mask)


def build(context):
    def wrapper(x):
        out = torch.empty_like(x)
        n_elements = x.numel()
        block_size = 1024
        if n_elements % block_size == 0:
            _abs_full_kernel[(n_elements // block_size,)](
                x, out, BLOCK_SIZE=block_size
            )
        else:
            _abs_tail_kernel[(triton.cdiv(n_elements, block_size),)](
                x, out, n_elements, BLOCK_SIZE=block_size
            )
        return out

    return wrapper
