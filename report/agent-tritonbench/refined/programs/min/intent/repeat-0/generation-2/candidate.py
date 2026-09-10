import math

import intent
import intent.language as I


@intent.fn
def _min_pair(lhs, rhs):
    lhs_value, lhs_index = lhs
    rhs_value, rhs_index = rhs

    # Keep the first NaN and otherwise select only a strictly smaller value.
    take_rhs = (lhs_value == lhs_value) & (
        (rhs_value != rhs_value) | (rhs_value < lhs_value)
    )
    return (
        I.select(take_rhs, rhs_value, lhs_value),
        I.select(take_rhs, rhs_index, lhs_index),
    )


@intent.kernel
def min_rows(
    input: I.In[I.f32, ("M", "N")],
    output: I.Out[I.f32, ("N",)],
    output_indices: I.Out[I.i64, ("N",)],
):
    row_axis = I.domain(0, input.shape[0])
    row_indices = I.indices(row_axis)
    row_indices = I.reshape(row_indices, (input.shape[0], 1))
    row_indices = I.cast(row_indices, I.i64)
    row_indices = row_indices + I.full(input.shape, fill=0, dtype=I.i64)

    value_identity = I.full(
        (input.shape[1],),
        fill=math.inf,
        dtype=I.f32,
    )
    index_identity = I.full(
        (input.shape[1],),
        fill=0,
        dtype=I.i64,
    )

    values, indices = I.reduce(
        (input, row_indices),
        axis=0,
        identity=(value_identity, index_identity),
        combine=_min_pair,
    )
    output[I.domain(0, output.shape[0])] = values
    output_indices[I.domain(0, output_indices.shape[0])] = indices


def build(context):
    compiled = context.compile("min_rows", min_rows)

    def wrapper(input, dim, keepdim=False, *, out=None):
        if dim != 0:
            raise ValueError("this specialization reduces dimension 0")

        if out is None:
            values, indices = compiled.run(input)
        else:
            values, indices = out
            if keepdim:
                compiled(input, values.squeeze(0), indices.squeeze(0))
            else:
                compiled(input, values, indices)

        if keepdim:
            if out is None:
                values = values.unsqueeze(0)
                indices = indices.unsqueeze(0)
        return values, indices

    return wrapper
