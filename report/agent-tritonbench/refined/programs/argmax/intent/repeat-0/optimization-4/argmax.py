import torch
import triton
import triton.language as tl


_REDUCE_BLOCK = 16384
_PARTIAL_BLOCK = 64


@triton.jit
def _argmax_stage1(input, partial_indices, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)

    values = tl.load(input + offsets)
    local_index = tl.argmax(values, axis=0, tie_break_left=True)
    best_index = block_start + local_index

    tl.store(partial_indices + pid, best_index)


@triton.jit
def _argmax_stage2(input, partial_indices, output, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    indices = tl.load(partial_indices + offsets)
    values = tl.load(input + indices)
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
        BLOCK=_REDUCE_BLOCK,
        num_warps=8,
    )
    _argmax_stage2[(1,)](
        input,
        partial_indices,
        output,
        BLOCK=_PARTIAL_BLOCK,
        num_warps=2,
    )


def run(input):
    output = torch.empty((), device=input.device, dtype=torch.int64)
    launch(input, output)
    return output
