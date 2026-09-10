import math

import intent
import intent.language as I


@intent.fn
def _minimum(lhs, rhs):
    return I.minimum(lhs, rhs)


@intent.fn
def _minimum_index(lhs, rhs):
    return I.minimum(lhs, rhs)


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
    values = I.reduce(
        input,
        axis=0,
        identity=value_identity,
        combine=_minimum,
    )

    is_nan_min = values != values
    is_match = (input == values) | (is_nan_min & (input != input))
    index_sentinel = I.full(input.shape, fill=input.shape[0], dtype=I.i64)
    matching_indices = I.select(is_match, row_indices, index_sentinel)

    index_identity = I.full(
        (input.shape[1],),
        fill=input.shape[0],
        dtype=I.i64,
    )
    indices = I.reduce(
        matching_indices,
        axis=0,
        identity=index_identity,
        combine=_minimum_index,
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
