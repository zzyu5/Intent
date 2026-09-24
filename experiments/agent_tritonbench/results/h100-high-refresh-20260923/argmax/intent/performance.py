import torch
import intent
import intent.language as I


GROUPS = 256
CHUNK = 4096


@intent.kernel
def _group_argmax(
    input_view: I.In[I.f16, (256, 4096)],
    partial_values: I.Out[I.f16, (256,)],
    partial_indices: I.Out[I.i64, (256,)],
):
    groups = I.domain(0, 256)
    columns = I.domain(0, 4096)
    for group in I.parallel(groups):
        row = input_view[group, columns]
        value, index = I.arg_reduce.max(row, axis=0)
        partial_values[group] = value
        partial_indices[group] = I.cast(index, I.i64)


@intent.kernel
def _final_argmax(
    partial_values: I.In[I.f16, (256,)],
    partial_indices: I.In[I.i64, (256,)],
    result: I.Out[I.i64, ()],
):
    groups = I.domain(0, 256)
    _, best_group = I.arg_reduce.max(partial_values[groups], axis=0)
    local_index = partial_indices[best_group]
    absolute_index = I.cast(best_group, I.i64) * 4096 + local_index
    result[()] = absolute_index


def build(context):
    group_argmax = context.compile("argmax_groups", _group_argmax, constexprs={})
    final_argmax = context.compile("argmax_final", _final_argmax, constexprs={})

    def argmax(input_tensor, dim, keepdim=False):
        input_view = input_tensor.reshape(GROUPS, CHUNK)
        partial_values = torch.empty(
            (GROUPS,), device=input_tensor.device, dtype=input_tensor.dtype
        )
        partial_indices = torch.empty(
            (GROUPS,), device=input_tensor.device, dtype=torch.int64
        )
        result = torch.empty((), device=input_tensor.device, dtype=torch.int64)

        group_argmax(input_view, partial_values, partial_indices)
        final_argmax(partial_values, partial_indices, result)

        if keepdim:
            return result.reshape(1)
        return result

    return argmax
