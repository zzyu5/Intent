import torch
import triton
import triton.language as tl


@triton.jit
def _add_sum_kernel(input_ptr, other_ptr, partial_ptr, n_elements, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    input_values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    other_values = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    values = input_values + alpha * other_values
    partial = tl.sum(values, axis=0)
    tl.store(partial_ptr + pid, partial)


@triton.jit
def _mean_partials_kernel(partial_ptr, output_ptr, n_partials, n_elements, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_partials
    partials = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(partials, axis=0)
    tl.store(output_ptr, total / n_elements)


def build(context):
    def wrapper(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
        count = input.numel()
        block = 1024
        n_partials = triton.cdiv(count, block)

        output_dtype = input.dtype if dtype is None else dtype
        partials = torch.empty((n_partials,), device=input.device, dtype=output_dtype)
        output = out if out is not None else torch.empty((), device=input.device, dtype=output_dtype)

        _add_sum_kernel[(n_partials,)](
            input,
            other,
            partials,
            count,
            alpha,
            BLOCK=block,
            num_warps=8,
        )
        _mean_partials_kernel[(1,)](
            partials,
            output,
            n_partials,
            count,
            BLOCK=block,
            num_warps=8,
        )
        return output

    return wrapper
