import torch
import triton
import triton.language as tl


@triton.jit
def sqrt_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    result = tl.sqrt(values)
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def wrapper(input, *, out=None):
        output = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        sqrt_kernel[(triton.cdiv(n_elements, 1024),)](
            input,
            output,
            n_elements,
            BLOCK_SIZE=1024,
            num_warps=4,
        )
        return output

    return wrapper
