import torch
import triton
import triton.language as tl


@triton.jit
def _logsumexp_partials(input_ptr, max_ptr, sum_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    values = tl.load(input_ptr + offsets)
    block_max = tl.max(values, axis=0)
    safe_max = tl.where(block_max == -float("inf"), 0.0, block_max)

    terms = tl.exp(values - safe_max)
    terms = tl.where(
        block_max == float("inf"),
        tl.where(values == float("inf"), 1.0, terms),
        terms,
    )
    block_sum = tl.sum(terms, axis=0)

    tl.store(max_ptr + pid, block_max)
    tl.store(sum_ptr + pid, block_sum)


@triton.jit
def _logsumexp_finalize(max_ptr, sum_ptr, output_ptr, n_partials, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)

    partial_max = tl.load(max_ptr + offsets)
    partial_sum = tl.load(sum_ptr + offsets)
    total_max = tl.max(partial_max, axis=0)
    safe_max = tl.where(total_max == -float("inf"), 0.0, total_max)

    terms = partial_sum * tl.exp(partial_max - safe_max)
    terms = tl.where(
        total_max == float("inf"),
        tl.where(partial_max == float("inf"), partial_sum, 0.0),
        terms,
    )
    total_sum = tl.sum(terms, axis=0)
    result = total_max + tl.log(total_sum)
    tl.store(output_ptr, result)


def build(context):
    del context

    def wrapper(input, dim, keepdim=False, *, out=None):
        if dim < 0:
            dim += input.ndim

        if out is None:
            shape = (1,) if keepdim else ()
            output = torch.empty(shape, device=input.device, dtype=input.dtype)
        else:
            output = out

        n_elements = input.numel()
        block_size = 1024
        n_partials = triton.cdiv(n_elements, block_size)
        partials = torch.empty(
            (2, n_partials), device=input.device, dtype=torch.float32
        )
        partial_max = partials[0]
        partial_sum = partials[1]

        _logsumexp_partials[(n_partials,)](
            input,
            partial_max,
            partial_sum,
            BLOCK=block_size,
            num_warps=4,
        )
        _logsumexp_finalize[(1,)](
            partial_max,
            partial_sum,
            output,
            n_partials,
            BLOCK=1024,
            num_warps=4,
        )
        return output

    return wrapper
