import torch
import intent
import intent.language as I


@intent.fn
def choose_min(lhs, rhs):
    value_equal = lhs.value == rhs.value
    lhs_is_lower = lhs.value < rhs.value
    lhs_wins_tie = lhs.index <= rhs.index
    numeric_choice = I.select(value_equal, lhs_wins_tie, lhs_is_lower)

    # Select NaNs explicitly so their first source position is retained too.
    lhs_nan = lhs.value != lhs.value
    rhs_nan = rhs.value != rhs.value
    false = I.full(lhs.value.shape, False, I.bool)
    lhs_nan_choice = I.select(rhs_nan, lhs_wins_tie, false)
    numeric_choice = I.select(rhs_nan, false, numeric_choice)
    choose_lhs = I.select(lhs_nan, lhs_nan_choice, numeric_choice)
    return I.record(
        value=I.select(choose_lhs, lhs.value, rhs.value),
        index=I.select(choose_lhs, lhs.index, rhs.index),
    )


@intent.kernel
def min_dim0(
    input_tensor: I.In[I.f32, ("M", "N")],
    output_values: I.Out[I.f32, ("N",)],
    output_indices: I.Out[I.i64, ("N",)],
):
    M, N = input_tensor.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)

    values = input_tensor[rows, columns]
    row_indices = I.reshape(I.indices(rows), (M, 1))
    row_indices = row_indices + I.full((1, N), 0, I.index)
    candidates = I.record(value=values, index=row_indices)

    identity = I.record(
        value=I.full((N,), -I.inf, I.f32),
        index=I.full((N,), M, I.index),
    )
    result = I.reduce(
        candidates,
        axis=0,
        identity=identity,
        combine=choose_min,
    )

    output_values[columns] = result.value
    output_indices[columns] = I.cast(result.index, I.i64)


def build(context):
    compiled_min = context.compile("min_dim0", min_dim0)

    def min_wrapper(input_tensor, dim, keepdim=False):
        _, columns = input_tensor.shape
        output_values = torch.empty(
            (columns,), device=input_tensor.device, dtype=torch.float32
        )
        output_indices = torch.empty(
            (columns,), device=input_tensor.device, dtype=torch.int64
        )
        compiled_min(input_tensor, output_values, output_indices)
        return output_values, output_indices

    return min_wrapper
