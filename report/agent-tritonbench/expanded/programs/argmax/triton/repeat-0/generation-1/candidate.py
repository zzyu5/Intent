import torch
import triton
import triton.language as tl


@triton.jit
def _argmax_chunks(
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
    nan_values = tl.isnan(values)
    numeric_values = tl.where(nan_values, -float("inf"), values)
    max_value = tl.max(numeric_values, axis=0)
    has_nan = tl.max(nan_values.to(tl.int32), axis=0) != 0

    first_max = tl.min(
        tl.where(valid & (nan_values == 0) & (values == max_value), offsets, n_elements),
        axis=0,
    )
    first_nan = tl.min(tl.where(valid & nan_values, offsets, n_elements), axis=0)
    winner = tl.where(has_nan, first_nan, first_max)

    winner_value = tl.load(input_ptr + winner)
    tl.store(partial_values_ptr + pid, winner_value)
    tl.store(partial_indices_ptr + pid, winner)


@triton.jit
def _argmax_partials(
    partial_values_ptr,
    partial_indices_ptr,
    output_ptr,
    n_partials,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    valid = offsets < n_partials
    values = tl.load(partial_values_ptr + offsets, mask=valid, other=-float("inf"))
    indices = tl.load(partial_indices_ptr + offsets, mask=valid, other=0)

    nan_values = tl.isnan(values)
    numeric_values = tl.where(nan_values, -float("inf"), values)
    max_value = tl.max(numeric_values, axis=0)
    has_nan = tl.max(nan_values.to(tl.int32), axis=0) != 0

    first_max = tl.min(
        tl.where(valid & (nan_values == 0) & (values == max_value), indices, n_partials),
        axis=0,
    )
    first_nan = tl.min(tl.where(valid & nan_values, indices, n_partials), axis=0)
    winner = tl.where(has_nan, first_nan, first_max)
    tl.store(output_ptr, winner.to(tl.int64))


def build(context):
    del context

    def wrapper(input, dim, keepdim=False):
        output_shape = (1,) if keepdim else ()
        output = torch.empty(output_shape, dtype=torch.int64, device=input.device)

        n_elements = input.numel()
        block = 1024
        n_partials = triton.cdiv(n_elements, block)
        partial_values = torch.empty((n_partials,), dtype=input.dtype, device=input.device)
        partial_indices = torch.empty((n_partials,), dtype=torch.int32, device=input.device)

        _argmax_chunks[(n_partials,)](
            input,
            partial_values,
            partial_indices,
            n_elements,
            BLOCK=block,
            num_warps=4,
        )
        _argmax_partials[(1,)](
            partial_values,
            partial_indices,
            output,
            n_partials,
            BLOCK=triton.next_power_of_2(n_partials),
            num_warps=4,
        )
        return output

    return wrapper
