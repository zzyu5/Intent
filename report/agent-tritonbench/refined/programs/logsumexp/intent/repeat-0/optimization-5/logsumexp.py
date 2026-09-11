import torch
import triton
import triton.language as tl


@triton.jit
def _partial_logsumexp(
    input_ptr,
    max_ptr,
    sum_ptr,
    BLOCK: tl.constexpr,
    LOG2E: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input_ptr + offsets)

    local_max = tl.max(values, axis=0)
    shifted = (values - local_max) * LOG2E
    exp_values = tl.exp2(shifted)
    # Avoid inf - inf producing NaN for inputs containing positive infinity.
    exp_values = tl.where(values == local_max, 1.0, exp_values)
    local_sum = tl.sum(exp_values, axis=0)

    tl.store(max_ptr + pid, local_max)
    tl.store(sum_ptr + pid, local_sum)


@triton.jit
def _finish_logsumexp(
    max_ptr,
    sum_ptr,
    output_ptr,
    BLOCK: tl.constexpr,
    LOG2E: tl.constexpr,
    LN2: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    partial_max = tl.load(max_ptr + offsets)
    global_max = tl.max(partial_max, axis=0)
    partial_sum = tl.load(sum_ptr + offsets)

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
    partial_max = torch.empty((128,), device=input.device, dtype=torch.float32)
    partial_sum = torch.empty((128,), device=input.device, dtype=torch.float32)

    _partial_logsumexp[(128,)](
        input,
        partial_max,
        partial_sum,
        BLOCK=8192,
        LOG2E=1.4426950408889634,
        num_warps=8,
        num_stages=2,
    )

    _finish_logsumexp[(1,)](
        partial_max,
        partial_sum,
        output,
        BLOCK=128,
        LOG2E=1.4426950408889634,
        LN2=0.6931471805599453,
        num_warps=1,
    )


def run(input):
    output = torch.empty((), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
