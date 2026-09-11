import torch
import triton
import triton.language as tl


@triton.jit
def _relu_fixed(input_ptr, output_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input_ptr + offsets)
    values = tl.maximum(values, 0.0, propagate_nan=tl.PropagateNan.ALL)
    tl.store(output_ptr + offsets, values)


@triton.jit
def _relu_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    values = tl.maximum(values, 0.0, propagate_nan=tl.PropagateNan.ALL)
    tl.store(output_ptr + offsets, values, mask=mask)


def build(context):
    def wrapper(input, inplace=False):
        n_elements = input.numel()
        output = input if inplace else torch.empty_like(input)
        if n_elements == 1048576:
            _relu_fixed[(512,)](input, output, BLOCK=2048, num_warps=16, num_stages=1)
            return output
        grid = (triton.cdiv(n_elements, 2048),)
        _relu_kernel[grid](input, output, n_elements, BLOCK=2048, num_warps=8, num_stages=1)
        return output

    return wrapper
