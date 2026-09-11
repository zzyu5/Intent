import torch
import triton
import triton.language as tl


@triton.jit
def _key_combine(key0, key1):
    return tl.where(key0 >= key1, key0, key1)


_PARTIAL_BLOCK = 1024
_NUM_PARTIALS = 1024


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def _partial_max(input, partial_values, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input + offsets, mask=mask, other=0.0)

    # Map float32 values to an unsigned monotonic key. NaNs sort above all
    # numbers, and both signed zeros share a key so the index field picks the
    # first zero exactly as torch.max does.
    bits = tl.cast(values, tl.uint32, bitcast=True)
    ordered = tl.where((bits >> 31) != 0, ~bits, bits ^ 0x80000000)
    ordered = tl.where(values == 0, 0x80000000, ordered)
    ordered = tl.where(values != values, 0xffffffff, ordered)

    index = tl.cast(offsets, tl.uint32)
    inverse_index = 0xffffffff - index
    key = (tl.cast(ordered, tl.uint64) << 32) | tl.cast(inverse_index, tl.uint64)
    key = tl.where(mask, key, 0)
    best = tl.reduce(key, axis=0, combine_fn=_key_combine)
    tl.store(partial_values + pid, best)


@triton.jit
def _final_max(input, partial_values, values_out, indices_out,
               n_partials, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_partials
    keys = tl.load(partial_values + offsets, mask=mask, other=0)
    key = tl.reduce(keys, axis=0, combine_fn=_key_combine)
    index = 0xffffffff - (key & 0xffffffff)
    value = tl.load(input + index)
    tl.store(values_out, value)
    tl.store(indices_out, index.to(tl.int64))


def launch(input, values, indices):
    n_elements = input.shape[0]
    partial_values = torch.empty((_NUM_PARTIALS,), device=input.device, dtype=torch.uint64)

    _partial_max[(_NUM_PARTIALS,)](input, partial_values, n_elements,
                                   BLOCK=_PARTIAL_BLOCK)
    _final_max[(1,)](input, partial_values, values, indices, _NUM_PARTIALS,
                     BLOCK=_NUM_PARTIALS)


def run(input):
    values = torch.empty((1,), device=input.device, dtype=input.dtype)
    indices = torch.empty((1,), device=input.device, dtype=torch.int64)
    launch(input, values, indices)
    return values, indices
