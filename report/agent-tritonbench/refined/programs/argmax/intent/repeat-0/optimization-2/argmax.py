import torch
import triton
import triton.language as tl


_REDUCE_BLOCK = 8192
_PARTIAL_BLOCK = 128


@triton.jit
def _argmax_stage1(input, partial_indices, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    values = tl.load(input + offsets, mask=mask, other=-float("inf"))
    local_index = tl.argmax(values, axis=0, tie_break_left=True)
    best_index = block_start + local_index

    tl.store(partial_indices + pid, best_index)


@triton.jit
def _argmax_stage2(input, partial_indices, output, n_partials, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_partials
    indices = tl.load(partial_indices + offsets, mask=mask, other=0)
    values = tl.load(input + indices, mask=mask, other=-float("inf"))
    partial_index = tl.argmax(values, axis=0, tie_break_left=True)
    best_index = tl.load(partial_indices + partial_index)
    tl.store(output, best_index.to(tl.int64))


def launch(input, output):
    n_elements = input.numel()
    n_partials = triton.cdiv(n_elements, _REDUCE_BLOCK)
    partial_indices = torch.empty((n_partials,), device=input.device, dtype=torch.int32)

    _argmax_stage1[(n_partials,)](
        input,
        partial_indices,
        n_elements,
        BLOCK=_REDUCE_BLOCK,
        num_warps=8,
    )
    _argmax_stage2[(1,)](
        input,
        partial_indices,
        output,
        n_partials,
        BLOCK=_PARTIAL_BLOCK,
        num_warps=4,
    )


def run(input):
    output = torch.empty((), device=input.device, dtype=torch.int64)
    launch(input, output)
    return output
