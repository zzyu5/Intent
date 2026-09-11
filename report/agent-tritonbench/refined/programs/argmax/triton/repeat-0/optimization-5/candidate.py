import torch
import triton
import triton.language as tl


@triton.jit
def _argmax_chunks(
    input_ptr,
    partial_indices_ptr,
    partial_values_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements

    values = tl.load(input_ptr + offsets, mask=valid, other=-float("inf"))
    values = values.to(tl.float32)
    nan_values = values != values
    positive_infinity = values == float("inf")
    # Float16 has no finite values above 65504.  Reserve 65505 for +inf so
    # NaNs can be represented as a distinct, higher-priority argmax value.
    key = tl.where(nan_values, float("inf"), tl.where(positive_infinity, 65505.0, values))
    local_index = tl.argmax(key, axis=0, tie_break_left=True)
    winner = pid * BLOCK + local_index
    winner_value = tl.load(input_ptr + winner, mask=winner < n_elements, other=-float("inf"))
    winner_value = winner_value.to(tl.float32)
    winner_nan = winner_value != winner_value
    winner_positive_infinity = winner_value == float("inf")
    winner_key = tl.where(
        winner_nan,
        float("inf"),
        tl.where(winner_positive_infinity, 65505.0, winner_value),
    )

    tl.store(partial_indices_ptr + pid, winner.to(tl.int32))
    tl.store(partial_values_ptr + pid, winner_key)


@triton.jit
def _argmax_partials(
    partial_indices_ptr,
    partial_values_ptr,
    output_ptr,
    n_partials,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    valid = offsets < n_partials
    values = tl.load(partial_values_ptr + offsets, mask=valid, other=-float("inf"))
    local_index = tl.argmax(values, axis=0, tie_break_left=True)
    winner = tl.load(partial_indices_ptr + local_index)
    tl.store(output_ptr, winner.to(tl.int64))


def build(context):
    del context

    def wrapper(input, dim, keepdim=False):
        output_shape = (1,) if keepdim else ()
        output = torch.empty(output_shape, dtype=torch.int64, device=input.device)

        n_elements = input.numel()
        block = 8192
        n_partials = triton.cdiv(n_elements, block)
        partial_indices = torch.empty((n_partials,), dtype=torch.int32, device=input.device)
        partial_values = torch.empty((n_partials,), dtype=torch.float32, device=input.device)

        _argmax_chunks[(n_partials,)](
            input,
            partial_indices,
            partial_values,
            n_elements,
            BLOCK=block,
            num_warps=16,
        )
        _argmax_partials[(1,)](
            partial_indices,
            partial_values,
            output,
            n_partials,
            BLOCK=triton.next_power_of_2(n_partials),
            num_warps=4,
        )
        return output

    return wrapper
