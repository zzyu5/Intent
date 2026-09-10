import torch
import triton
import triton.language as tl


@triton.jit
def _exp_sum_chunks(
    input_ptr,
    partial_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    terms = tl.exp(values)
    partial = tl.sum(terms, axis=0)
    tl.store(partial_ptr + pid, partial)


@triton.jit
def _exp_mean_finalize(
    partial_ptr,
    output_ptr,
    n_partials,
    inv_count,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_partials
    partials = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(partials, axis=0)
    tl.store(output_ptr, total * inv_count)


def build(context):
    del context

    def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None):
        del dim, keepdim, dtype
        n_elements = input.numel()
        block = 1024
        n_partials = triton.cdiv(n_elements, block)
        partials = torch.empty(
            (n_partials,), device=input.device, dtype=torch.float32
        )
        output = out
        if output is None:
            output = torch.empty((), device=input.device, dtype=torch.float32)

        _exp_sum_chunks[(n_partials,)](
            input, partials, n_elements, BLOCK=block, num_warps=8
        )
        _exp_mean_finalize[(1,)](
            partials,
            output,
            n_partials,
            1.0 / n_elements,
            BLOCK=1024,
            num_warps=8,
        )
        return output

    return exp_mean
