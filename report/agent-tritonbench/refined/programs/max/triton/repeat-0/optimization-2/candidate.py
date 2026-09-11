import collections

import torch
import triton
import triton.language as tl


MaxResult = collections.namedtuple("max", ["values", "indices"])

_MAX_PARTIALS = 256


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 4096}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 8192}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 8192}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 8192}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK": 16384}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 16384}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 16384}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK": 32768}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 32768}, num_warps=16, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def _max_partial_kernel(
    input_ptr,
    partial_values_ptr,
    partial_indices_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=valid, other=-float("inf"))
    best = tl.argmax(values, axis=0, tie_break_left=True)
    best_offset = pid * BLOCK + best
    best_value = tl.load(input_ptr + best_offset)
    tl.store(partial_values_ptr + pid, best_value)
    tl.store(partial_indices_ptr + pid, best_offset)

    tail_offsets = tl.arange(0, _MAX_PARTIALS)
    tail_mask = (pid == 0) & (tail_offsets >= (n_elements + BLOCK - 1) // BLOCK)
    tl.store(partial_values_ptr + tail_offsets, -float("inf"), mask=tail_mask)
    tl.store(partial_indices_ptr + tail_offsets, 0, mask=tail_mask)


@triton.jit
def _max_final_kernel(
    partial_values_ptr,
    partial_indices_ptr,
    output_value_ptr,
    output_index_ptr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(partial_values_ptr + offsets)
    best = tl.argmax(values, axis=0, tie_break_left=True)
    best_value = tl.load(partial_values_ptr + best)
    best_index = tl.load(partial_indices_ptr + best)
    tl.store(output_value_ptr, best_value)
    tl.store(output_index_ptr, best_index.to(tl.int64))


def build(context):
    def wrapper(input, dim, keepdim=False, *, out=None):
        if dim not in (0, -1):
            raise ValueError("the fixed reduction only supports dim 0")

        n_elements = input.numel()

        if out is None:
            output_shape = (1,) if keepdim else ()
            output_values = torch.empty(output_shape, dtype=input.dtype, device=input.device)
            output_indices = torch.empty(output_shape, dtype=torch.int64, device=input.device)
        else:
            output_values, output_indices = out

        partial_values = torch.empty(
            (_MAX_PARTIALS,), dtype=input.dtype, device=input.device
        )
        partial_indices = torch.empty(
            (_MAX_PARTIALS,), dtype=torch.int32, device=input.device
        )

        _max_partial_kernel[lambda META: (triton.cdiv(n_elements, META["BLOCK"]),)](
            input,
            partial_values,
            partial_indices,
            n_elements,
        )
        _max_final_kernel[(1,)](
            partial_values,
            partial_indices,
            output_values,
            output_indices,
            BLOCK=_MAX_PARTIALS,
            num_warps=1,
        )
        return MaxResult(output_values, output_indices)

    return wrapper
