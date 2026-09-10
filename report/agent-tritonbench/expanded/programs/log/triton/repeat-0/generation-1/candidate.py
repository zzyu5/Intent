import torch
import triton
import triton.language as tl


@triton.jit
def log_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, tl.log(values), mask=mask)


def build(context):
    def wrapper(input, *, out=None):
        if out is None:
            out = torch.empty_like(input)
        n_elements = input.numel()
        log_kernel[(triton.cdiv(n_elements, 256),)](
            input,
            out,
            n_elements,
            BLOCK_SIZE=256,
        )
        return out

    return wrapper
