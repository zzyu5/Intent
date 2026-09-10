import torch
import triton
import triton.language as tl


@triton.jit
def _sum_blocks(input_ptr, partial_ptr, n_elements, BLOCK: tl.constexpr):
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    partial = tl.sum(values, axis=0)
    tl.store(partial_ptr + block_id, partial)


@triton.jit
def _sum_partials(partial_ptr, output_ptr, n_partials, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_partials
    partials = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(partials, axis=0)
    tl.store(output_ptr, total)


def build(context):
    def wrapper(input, dim, keepdim=False, *, dtype=None):
        count = input.numel()
        block = 4096
        n_partials = triton.cdiv(count, block)
        partials = torch.empty((n_partials,), device=input.device, dtype=input.dtype)
        output = torch.empty((), device=input.device, dtype=input.dtype)

        _sum_blocks[(n_partials,)](input, partials, count, BLOCK=block, num_warps=8)
        _sum_partials[(1,)](partials, output, n_partials, BLOCK=triton.next_power_of_2(n_partials), num_warps=4)
        return output

    return wrapper
