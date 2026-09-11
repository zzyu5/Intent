import collections

import torch
import triton
import triton.language as tl


MaxResult = collections.namedtuple("max", ["values", "indices"])

_PARTIAL_STRIDE = 256


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
    partial_buffer_ptr,
    n_elements,
    PARTIAL_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input_ptr + offsets)
    best = tl.argmax(values, axis=0, tie_break_left=True)
    best_offset = pid * BLOCK + best
    best_value = tl.load(input_ptr + best_offset)
    tl.store(partial_buffer_ptr + pid, best_value)
    tl.store(
        partial_buffer_ptr + PARTIAL_STRIDE + pid,
        best_offset.to(tl.float32),
    )
    n_partials = (n_elements + BLOCK - 1) // BLOCK
    tl.store(
        partial_buffer_ptr + 2 * PARTIAL_STRIDE,
        n_partials.to(tl.float32),
        mask=pid == 0,
    )


@triton.jit
def _max_final_kernel(
    partial_buffer_ptr,
    output_value_ptr,
    output_index_ptr,
    PARTIAL_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    n_partials = tl.load(partial_buffer_ptr + 2 * PARTIAL_STRIDE).to(tl.int32)
    offsets = tl.arange(0, BLOCK)
    valid = offsets < n_partials
    values = tl.load(
        partial_buffer_ptr + offsets,
        mask=valid,
        other=-float("inf"),
    )
    best = tl.argmax(values, axis=0, tie_break_left=True)
    best_value = tl.load(partial_buffer_ptr + best)
    best_index = tl.load(partial_buffer_ptr + PARTIAL_STRIDE + best)
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

        partial_buffer = torch.empty(
            (2 * _PARTIAL_STRIDE + 1,),
            dtype=input.dtype,
            device=input.device,
        )
        _max_partial_kernel[
            lambda META: (triton.cdiv(n_elements, META["BLOCK"]),)
        ](
            input,
            partial_buffer,
            n_elements,
            PARTIAL_STRIDE=_PARTIAL_STRIDE,
        )
        _max_final_kernel[(1,)](
            partial_buffer,
            output_values,
            output_indices,
            PARTIAL_STRIDE=_PARTIAL_STRIDE,
            BLOCK=_PARTIAL_STRIDE,
            num_warps=1,
        )
        return MaxResult(output_values, output_indices)

    return wrapper
