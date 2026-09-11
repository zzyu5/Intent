import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 8192}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 8192}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK": 8192}, num_warps=32, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def _partial_logsumexp(
    input_ptr,
    max_ptr,
    sum_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    LOG2E: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))

    local_max = tl.max(values, axis=0)
    shifted = (values - local_max) * LOG2E
    exp_values = tl.exp2(shifted)
    # Avoid inf - inf producing NaN for inputs containing positive infinity.
    exp_values = tl.where(values == local_max, 1.0, exp_values)
    exp_values = tl.where(mask, exp_values, 0.0)
    local_sum = tl.sum(exp_values, axis=0)

    tl.store(max_ptr + pid, local_max)
    tl.store(sum_ptr + pid, local_sum)


@triton.jit
def _finish_logsumexp(
    max_ptr,
    sum_ptr,
    output_ptr,
    n_partials,
    BLOCK: tl.constexpr,
    LOG2E: tl.constexpr,
    LN2: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_partials
    partial_max = tl.load(max_ptr + offsets, mask=mask, other=-float("inf"))
    global_max = tl.max(partial_max, axis=0)
    partial_sum = tl.load(sum_ptr + offsets, mask=mask, other=0.0)

    scaled_sum = tl.where(
        partial_max == global_max,
        partial_sum,
        tl.where(
            partial_max != -float("inf"),
            partial_sum * tl.exp2((partial_max - global_max) * LOG2E),
            0.0,
        ),
    )
    total = tl.sum(scaled_sum, axis=0)
    result = global_max + tl.log2(total) * LN2
    tl.store(output_ptr, result)


def launch(input, output):
    n_elements = input.shape[0]

    n_partials = triton.cdiv(n_elements, 8192)
    partial_max = torch.empty((n_partials,), device=input.device, dtype=torch.float32)
    partial_sum = torch.empty((n_partials,), device=input.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK"]),)
    _partial_logsumexp[grid](
        input,
        partial_max,
        partial_sum,
        n_elements,
        LOG2E=1.4426950408889634,
    )

    _finish_logsumexp[(1,)](
        partial_max,
        partial_sum,
        output,
        n_partials,
        BLOCK=128,
        LOG2E=1.4426950408889634,
        LN2=0.6931471805599453,
        num_warps=4,
    )


def run(input):
    output = torch.empty((), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
