import torch
import intent
import intent.language as I


GROUPS = 1024
CHUNK = 1024
ELEMENTS = GROUPS * CHUNK


@intent.kernel
def _partial_argmax(
    x: I.In[I.f32, (GROUPS, CHUNK)],
    partial_values: I.Out[I.f32, (GROUPS,)],
    partial_indices: I.Out[I.i64, (GROUPS,)],
):
    members = I.domain(0, CHUNK)
    for group in I.parallel(I.domain(0, GROUPS)):
        values = x[group, members]
        value, local_index = I.arg_reduce.max(values, axis=0)
        partial_values[group] = value
        partial_indices[group] = I.cast(group * CHUNK + local_index, I.i64)


@intent.kernel
def _final_argmax(
    partial_values: I.In[I.f32, (GROUPS,)],
    partial_indices: I.In[I.i64, (GROUPS,)],
    value_out: I.Out[I.f32, ()],
    index_out: I.Out[I.i64, ()],
):
    value, winning_group = I.arg_reduce.max(partial_values, axis=0)
    value_out[()] = value
    index_out[()] = partial_indices[winning_group]


def build(context):
    partial_artifact = context.compile("max_partial_argmax", _partial_argmax)
    final_artifact = context.compile("max_final_argmax", _final_argmax)

    def max(input, dim, keepdim=False):
        input_2d = input.reshape(GROUPS, CHUNK)
        partial_values = torch.empty(
            (GROUPS,), device=input.device, dtype=torch.float32
        )
        partial_indices = torch.empty(
            (GROUPS,), device=input.device, dtype=torch.int64
        )
        value_out = torch.empty((), device=input.device, dtype=torch.float32)
        index_out = torch.empty((), device=input.device, dtype=torch.int64)

        partial_artifact(input_2d, partial_values, partial_indices)
        final_artifact(
            partial_values,
            partial_indices,
            value_out,
            index_out,
        )

        if keepdim:
            return value_out.reshape(1), index_out.reshape(1)
        return value_out, index_out

    return max
