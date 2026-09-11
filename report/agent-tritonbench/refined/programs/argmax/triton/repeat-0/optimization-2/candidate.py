import torch
import triton
import triton.language as tl


@triton.jit
def _argmax_chunks(
    input_ptr,
    partial_indices_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements

    values = tl.load(input_ptr + offsets, mask=valid, other=-float("inf"))
    nan_values = values != values
    values = tl.where(nan_values, -float("inf"), values)
    max_value = tl.max(values, axis=0)
    has_nan = tl.max(nan_values.to(tl.int32), axis=0) != 0

    winner_mask = tl.where(has_nan, nan_values, values == max_value)
    winner = tl.min(tl.where(valid & winner_mask, offsets, n_elements), axis=0)

    tl.store(partial_indices_ptr + pid, winner)


@triton.jit
def _argmax_partials(
    partial_indices_ptr,
    input_ptr,
    output_ptr,
    n_partials,
    n_elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    valid = offsets < n_partials
    indices = tl.load(partial_indices_ptr + offsets, mask=valid, other=0)
    values = tl.load(input_ptr + indices, mask=valid, other=-float("inf"))

    nan_values = values != values
    values = tl.where(nan_values, -float("inf"), values)
    max_value = tl.max(values, axis=0)
    has_nan = tl.max(nan_values.to(tl.int32), axis=0) != 0

    winner_mask = tl.where(has_nan, nan_values, values == max_value)
    winner = tl.min(tl.where(valid & winner_mask, indices, n_elements), axis=0)
    tl.store(output_ptr, winner.to(tl.int64))


def build(context):
    del context

    def wrapper(input, dim, keepdim=False):
        output_shape = (1,) if keepdim else ()
        output = torch.empty(output_shape, dtype=torch.int64, device=input.device)

        n_elements = input.numel()
        block = 2048
        n_partials = triton.cdiv(n_elements, block)
        partial_indices = torch.empty((n_partials,), dtype=torch.int32, device=input.device)

        _argmax_chunks[(n_partials,)](
            input,
            partial_indices,
            n_elements,
            BLOCK=block,
            num_warps=8,
        )
        _argmax_partials[(1,)](
            partial_indices,
            input,
            output,
            n_partials,
            n_elements,
            BLOCK=triton.next_power_of_2(n_partials),
            num_warps=4,
        )
        return output

    return wrapper
