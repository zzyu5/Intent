import torch
import triton
import triton.language as tl


@triton.jit
def _argmax_combine(value0, index0, value1, index1):
    value0_is_nan = value0 != value0
    value1_is_nan = value1 != value1
    take0 = (value0 > value1) | ((value0 == value1) & (index0 <= index1))
    take0 = tl.where(value0_is_nan, tl.where(value1_is_nan, index0 <= index1, True),
                     tl.where(value1_is_nan, False, take0))
    return tl.where(take0, value0, value1), tl.where(take0, index0, index1)


_PARTIAL_BLOCK = 2048
_NUM_PARTIALS = 512


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def _partial_max(input, partial_values, partial_indices, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input + offsets)
    value, index = tl.reduce((values, offsets), axis=0, combine_fn=_argmax_combine)
    tl.store(partial_values + pid, value)
    tl.store(partial_indices + pid, index)


@triton.jit
def _final_max(partial_values, partial_indices, values_out, indices_out,
               BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(partial_values + offsets)
    indices = tl.load(partial_indices + offsets)
    value, index = tl.reduce((values, indices), axis=0, combine_fn=_argmax_combine)
    tl.store(values_out, value)
    tl.store(indices_out, index.to(tl.int64))


def launch(input, values, indices):
    n_elements = input.shape[0]
    partial_values = torch.empty((_NUM_PARTIALS,), device=input.device, dtype=input.dtype)
    partial_indices = torch.empty((_NUM_PARTIALS,), device=input.device, dtype=torch.int32)

    _partial_max[(_NUM_PARTIALS,)](input, partial_values, partial_indices,
                                   n_elements, BLOCK=_PARTIAL_BLOCK)
    _final_max[(1,)](partial_values, partial_indices, values, indices,
                     BLOCK=_NUM_PARTIALS)


def run(input):
    values = torch.empty((1,), device=input.device, dtype=input.dtype)
    indices = torch.empty((1,), device=input.device, dtype=torch.int64)
    launch(input, values, indices)
    return values, indices
