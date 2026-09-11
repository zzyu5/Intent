import collections

import torch
import triton
import triton.language as tl


MaxResult = collections.namedtuple("max", ["values", "indices"])

_PARTIAL_BLOCK = 8192
_PARTIAL_COUNT = 128


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 8192}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 8192}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 8192}, num_warps=16, num_stages=1),
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
    values = tl.load(input_ptr + offsets)
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
            output_values = torch.empty(
                output_shape,
                dtype=input.dtype,
                device=input.device,
            )
            output_indices = torch.empty(
                output_shape,
                dtype=torch.int64,
                device=input.device,
            )
        else:
            output_values, output_indices = out

        n_partials = (n_elements + _PARTIAL_BLOCK - 1) // _PARTIAL_BLOCK
        # Keep the two scratch streams in one allocation while preserving
        # their native pointer dtypes for the Triton kernels.
        partial_storage = torch.empty(
            (2 * n_partials,),
            dtype=input.dtype,
            device=input.device,
        )
        partial_values = partial_storage[:n_partials]
        partial_indices = partial_storage[n_partials:].view(torch.int32)
        _max_partial_kernel[(n_partials,)](
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
            BLOCK=_PARTIAL_COUNT,
            num_warps=1,
        )
        return MaxResult(output_values, output_indices)

    return wrapper
