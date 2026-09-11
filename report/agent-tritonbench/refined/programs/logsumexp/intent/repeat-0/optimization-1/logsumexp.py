import torch
import triton
import triton.language as tl


LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


@triton.jit
def _max_combine(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 512, "CHUNKS": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 512, "CHUNKS": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 512, "CHUNKS": 16}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK": 1024, "CHUNKS": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 1024, "CHUNKS": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 1024, "CHUNKS": 8}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK": 2048, "CHUNKS": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 2048, "CHUNKS": 4}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 2048, "CHUNKS": 4}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK": 4096, "CHUNKS": 2}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 4096, "CHUNKS": 2}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 4096, "CHUNKS": 2}, num_warps=16, num_stages=2),
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
    CHUNKS: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * (BLOCK * CHUNKS)
    running_max = -float("inf")
    running_sum = 0.0

    for chunk in range(CHUNKS):
        offsets = base + chunk * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        values = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
        chunk_max = tl.reduce(values, axis=0, combine_fn=_max_combine)

        shifted = (values - chunk_max) * LOG2E
        exp_values = tl.exp2(shifted)
        exp_values = tl.where(values == chunk_max, 1.0, exp_values)
        exp_values = tl.where(mask & (chunk_max != -float("inf")), exp_values, 0.0)
        chunk_sum = tl.sum(exp_values, axis=0)

        new_max = tl.maximum(running_max, chunk_max, propagate_nan=tl.PropagateNan.ALL)
        old_scale = tl.where(
            (running_max == new_max) & (running_max != -float("inf")),
            1.0,
            tl.where(
                running_max != -float("inf"),
                tl.exp2((running_max - new_max) * LOG2E),
                0.0,
            ),
        )
        chunk_scale = tl.where(
            (chunk_max == new_max) & (chunk_max != -float("inf")),
            1.0,
            tl.where(
                chunk_max != -float("inf"),
                tl.exp2((chunk_max - new_max) * LOG2E),
                0.0,
            ),
        )
        running_sum = running_sum * old_scale + chunk_sum * chunk_scale
        running_max = new_max

    tl.store(max_ptr + pid, running_max)
    tl.store(sum_ptr + pid, running_sum)


@triton.jit
def _finish_logsumexp(max_ptr, sum_ptr, output_ptr, n_partials, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_partials
    partial_max = tl.load(max_ptr + offsets, mask=mask, other=-float("inf"))
    global_max = tl.reduce(partial_max, axis=0, combine_fn=_max_combine)
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
    result = global_max + tl.log(total) * LN2
    tl.store(output_ptr, result)


def launch(input, output):
    n_elements = input.shape[0]
    elements_per_program = 8192
    n_partials = triton.cdiv(n_elements, elements_per_program)
    partial_max = torch.empty((n_partials,), device=input.device, dtype=torch.float32)
    partial_sum = torch.empty((n_partials,), device=input.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK"] * META["CHUNKS"]),)
    _partial_logsumexp[grid](input, partial_max, partial_sum, n_elements)
    _finish_logsumexp[(1,)](partial_max, partial_sum, output, n_partials, BLOCK=128, num_warps=4)


def run(input):
    output = torch.empty((), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
