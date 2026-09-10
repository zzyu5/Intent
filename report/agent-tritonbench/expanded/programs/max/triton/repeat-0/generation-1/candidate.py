import collections

import torch
import triton
import triton.language as tl


MaxResult = collections.namedtuple("max", ["values", "indices"])


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
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
    best = tl.argmax(values, axis=0, tie_break_left=True)
    best_offset = pid * BLOCK + best
    best_value = tl.load(input_ptr + best_offset)
    tl.store(partial_values_ptr + pid, best_value)
    tl.store(partial_indices_ptr + pid, best_offset)


@triton.jit
def _max_final_kernel(
    partial_values_ptr,
    partial_indices_ptr,
    output_value_ptr,
    output_index_ptr,
    n_partials,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_partials
    values = tl.load(partial_values_ptr + offsets, mask=mask, other=-float("inf"))
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
        block = 4096
        n_partials = (n_elements + block - 1) // block

        if out is None:
            output_shape = (1,) if keepdim else ()
            output_values = torch.empty(output_shape, dtype=input.dtype, device=input.device)
            output_indices = torch.empty(output_shape, dtype=torch.int64, device=input.device)
        else:
            output_values, output_indices = out

        partial_values = torch.empty(
            (n_partials,), dtype=input.dtype, device=input.device
        )
        partial_indices = torch.empty(
            (n_partials,), dtype=torch.int32, device=input.device
        )

        _max_partial_kernel[(n_partials,)](
            input,
            partial_values,
            partial_indices,
            n_elements,
            BLOCK=block,
            num_warps=8,
        )
        _max_final_kernel[(1,)](
            partial_values,
            partial_indices,
            output_values,
            output_indices,
            n_partials,
            BLOCK=256,
            num_warps=8,
        )
        return MaxResult(output_values, output_indices)

    return wrapper
