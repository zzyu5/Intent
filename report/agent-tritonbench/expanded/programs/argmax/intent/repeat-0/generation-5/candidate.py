import torch
import intent
import intent.language as I


@intent.kernel
def argmax_partials(
    input: I.In[I.f16, (16, 65536)],
    partial_values: I.Out[I.f16, (16,)],
    partial_indices: I.Out[I.i64, (16,)],
):
    identity = I.cast(-65504, I.f16)
    result = I.arg_reduce.max(input, axis=1, identity=identity)
    parts = I.domain(0, 16)

    for part in I.parallel(parts):
        partial_values[part] = result[0][part]
        partial_indices[part] = I.cast(result[1][part], I.i64)


@intent.kernel
def argmax_final(
    partial_values: I.In[I.f16, (16,)],
    partial_indices: I.In[I.i64, (16,)],
    output: I.Out[I.i64, (1,)],
    CHUNK: I.Constexpr[int],
):
    identity = I.cast(-65504, I.f16)
    result = I.arg_reduce.max(partial_values, axis=0, identity=identity)
    winner = I.cast(result[1], I.i64)
    output[0] = winner * CHUNK + partial_indices[winner]


def build(context):
    partials = context.compile(
        "argmax_partials_16x65536",
        argmax_partials,
        constexprs={},
    )
    final = context.compile(
        "argmax_final_16",
        argmax_final,
        constexprs={"CHUNK": 65536},
    )

    def wrapper(input, dim, keepdim=False):
        input_2d = input.view(16, 65536)
        partial_values = torch.empty((16,), device=input.device, dtype=input.dtype)
        partial_indices = torch.empty((16,), device=input.device, dtype=torch.int64)
        partials(input_2d, partial_values, partial_indices)

        output = torch.empty((1,), device=input.device, dtype=torch.int64)
        final(partial_values, partial_indices, output)
        return output.view(())

    return wrapper
