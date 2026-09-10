import torch
import triton
import triton.language as tl


@triton.jit
def _std_partial_moments(
    input_ptr,
    mean_ptr,
    m2_ptr,
    count_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    count = tl.sum(mask.to(tl.float32), axis=0)
    mean = tl.sum(values, axis=0) / count
    delta = values - mean
    m2 = tl.sum(tl.where(mask, delta * delta, 0.0), axis=0)

    tl.store(mean_ptr + pid, mean)
    tl.store(m2_ptr + pid, m2)
    tl.store(count_ptr + pid, count)


@triton.jit
def _std_finalize(
    mean_ptr,
    m2_ptr,
    count_ptr,
    output_ptr,
    n_partials,
    correction,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_partials

    means = tl.load(mean_ptr + offsets, mask=mask, other=0.0)
    m2s = tl.load(m2_ptr + offsets, mask=mask, other=0.0)
    counts = tl.load(count_ptr + offsets, mask=mask, other=0.0)

    total_count = tl.sum(tl.where(mask, counts, 0.0), axis=0)
    weighted_mean = tl.sum(tl.where(mask, means * counts, 0.0), axis=0)
    mean = weighted_mean / total_count
    delta = means - mean
    total_m2 = tl.sum(tl.where(mask, m2s + counts * delta * delta, 0.0), axis=0)

    denominator = tl.maximum(total_count - correction, 0.0)
    result = tl.sqrt(total_m2 / denominator)
    tl.store(output_ptr, result)


def build(context):
    def wrapper(input, dim=None, *, correction=1, keepdim=False, out=None):
        n_elements = input.numel()
        block = 1024
        n_partials = triton.cdiv(n_elements, block)

        if out is None:
            output_shape = (1,) if keepdim else ()
            output = torch.empty(output_shape, device=input.device, dtype=input.dtype)
        else:
            output = out

        partials = torch.empty((3, n_partials), device=input.device, dtype=torch.float32)
        grid = (n_partials,)
        _std_partial_moments[grid](
            input,
            partials[0],
            partials[1],
            partials[2],
            n_elements,
            BLOCK=block,
        )
        _std_finalize[(1,)](
            partials[0],
            partials[1],
            partials[2],
            output,
            n_partials,
            correction,
            BLOCK=block,
        )
        return output

    return wrapper
