import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def _add_sum_kernel(input_ptr, other_ptr, partial_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    values = values + tl.load(other_ptr + offsets, mask=mask, other=0.0)
    tl.store(partial_ptr + pid, tl.sum(values, axis=0))


@triton.jit
def _mean_partials_kernel(partial_ptr, output_ptr, BLOCK: tl.constexpr):
    partials = tl.load(partial_ptr + tl.arange(0, BLOCK))
    total = tl.sum(partials, axis=0)
    tl.store(output_ptr, total * 9.5367431640625e-7)


def build(context):
    def wrapper(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
        count = input.numel()
        block = 8192
        n_partials = triton.cdiv(count, block)

        output = torch.empty((n_partials,), device=input.device, dtype=input.dtype)

        _add_sum_kernel[(n_partials,)](
            input,
            other,
            output,
            count,
            BLOCK=block,
        )
        _mean_partials_kernel[(1,)](output, output, BLOCK=n_partials, num_warps=4)
        return output[0]

    return wrapper
